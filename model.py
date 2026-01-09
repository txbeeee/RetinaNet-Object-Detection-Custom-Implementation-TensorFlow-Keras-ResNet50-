def build_backbone():
    backbone = ResNet50(include_top=False, weights="imagenet")
    for layer in backbone.layers[:100]:
        layer.trainable = False

    c3 = backbone.get_layer("conv3_block4_out").output
    c4 = backbone.get_layer("conv4_block6_out").output
    c5 = backbone.get_layer("conv5_block3_out").output

    p5 = layers.Conv2D(256, 1, name="C5")(c5)
    p5_up = layers.UpSampling2D()(p5)

    p4 = layers.Conv2D(256, 1, name="C4")(c4)
    p4 = layers.Add()([p4, p5_up])
    p4_up = layers.UpSampling2D()(p4)

    p3 = layers.Conv2D(256, 1, name="C3")(c3)
    p3 = layers.Add()([p3, p4_up])

    p3 = layers.Conv2D(256, 3, padding="same", activation="relu")(p3)
    p4 = layers.Conv2D(256, 3, padding="same", activation="relu")(p4)
    p5 = layers.Conv2D(256, 3, padding="same", activation="relu")(p5)

    return models.Model(inputs=backbone.input, outputs=[p3, p4, p5])

class SimpleRetinaNet(tf.keras.Model):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        self.anchor_box = AnchorBox(
            strides=(4, 8, 16),
            sizes=(32, 64, 128),
            ratios=(0.5, 1.0, 2.0),
            scales=(1.0, 2 ** (1/3), 2 ** (2/3))
        )
        self.backbone = build_backbone()
        self.num_anchors = len(self.anchor_box.ratios) * len(self.anchor_box.scales)
        print(f"Model config: anchors_per_location={self.num_anchors}, classes={num_classes}")

        bias_init = tf.constant_initializer(-4.595)
        self.cls_head = tf.keras.Sequential([
            layers.Conv2D(256, 3, padding='same', activation='relu'),
            layers.Conv2D(256, 3, padding='same', activation='relu'),
            layers.Conv2D(self.num_anchors * num_classes, 3, padding='same',
                          bias_initializer=bias_init)  
        ])
        self.box_head = tf.keras.Sequential([
            layers.Conv2D(256, 3, padding='same', activation='relu'),
            layers.Conv2D(256, 3, padding='same', activation='relu'),
            layers.Conv2D(self.num_anchors * 4, 3, padding='same')
        ])
        self._precomputed_anchors = None

    def build(self, input_shape):
        super().build(input_shape)
        dummy = tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3], dtype=tf.float32)
        features = self.backbone(dummy, training=False)
        feature_shapes = []
        for feat in features:
            h = int(feat.shape[1])
            w = int(feat.shape[2])
            feature_shapes.append((h, w))
        self._precomputed_anchors = self.anchor_box.from_feature_maps(feature_shapes)
        self._precomputed_anchors = tf.cast(self._precomputed_anchors, tf.float32)

    def _get_anchors(self, batch_size):
        if self._precomputed_anchors is None:
            raise RuntimeError("Anchors not initialized. Ensure build() or call() computed anchors.")
        anchors = tf.expand_dims(self._precomputed_anchors, 0)
        return tf.tile(anchors, [batch_size, 1, 1])
    
    def call(self, images, training=False):
        features = self.backbone(images, training=training)
        cls_outputs = []
        box_outputs = []

        for feat in features:
            B = tf.shape(feat)[0]
            H = tf.shape(feat)[1]
            W = tf.shape(feat)[2]

            cls_out = self.cls_head(feat)
            box_out = self.box_head(feat)

            cls_out = tf.reshape(cls_out, [B, H * W * self.num_anchors, self.num_classes])
            box_out = tf.reshape(box_out, [B, H * W * self.num_anchors, 4])

            cls_outputs.append(cls_out)
            box_outputs.append(box_out)

        cls_pred = tf.concat(cls_outputs, axis=1)
        box_pred = tf.concat(box_outputs, axis=1)
        return cls_pred, box_pred

    def _encode_boxes(self, boxes, anchors):
        anchor_width  = anchors[..., 2] - anchors[..., 0]
        anchor_height = anchors[..., 3] - anchors[..., 1]
        anchor_cx = (anchors[..., 0] + anchors[..., 2]) / 2.0
        anchor_cy = (anchors[..., 1] + anchors[..., 3]) / 2.0

        box_width  = boxes[..., 2] - boxes[..., 0]
        box_height = boxes[..., 3] - boxes[..., 1]
        box_cx = (boxes[..., 0] + boxes[..., 2]) / 2.0
        box_cy = (boxes[..., 1] + boxes[..., 3]) / 2.0

        dx = (box_cx - anchor_cx) / anchor_width
        dy = (box_cy - anchor_cy) / anchor_height
        dw = tf.math.log(box_width  / anchor_width)
        dh = tf.math.log(box_height / anchor_height)

        return tf.stack([dx, dy, dw, dh], axis=-1)


    def _decode_boxes(self, encoded_boxes, anchors):
        anchor_width  = anchors[..., 2] - anchors[..., 0]
        anchor_height = anchors[..., 3] - anchors[..., 1]
        anchor_cx = (anchors[..., 0] + anchors[..., 2]) / 2.0
        anchor_cy = (anchors[..., 1] + anchors[..., 3]) / 2.0

        dx = encoded_boxes[..., 0]
        dy = encoded_boxes[..., 1]
        dw = encoded_boxes[..., 2]
        dh = encoded_boxes[..., 3]

        pred_cx = anchor_cx + dx * anchor_width
        pred_cy = anchor_cy + dy * anchor_height
        pred_width  = anchor_width  * tf.exp(dw)
        pred_height = anchor_height * tf.exp(dh)

        x1 = pred_cx - pred_width  / 2.0
        y1 = pred_cy - pred_height / 2.0
        x2 = pred_cx + pred_width  / 2.0
        y2 = pred_cy + pred_height / 2.0

        return tf.stack([x1, y1, x2, y2], axis=-1)

    def _compute_loss(self, cls_pred, box_pred, anchors, targets):
        boxes = targets["boxes"]  
        classes = targets["classes"] 
        B = tf.shape(boxes)[0]
        pos_iou_thr = 0.5
        neg_iou_thr = 0.4

        def process_single_element(i):
            cls_pred_i = cls_pred[i]     
            box_pred_i = box_pred[i]     
            anchors_i = anchors[i]      
            boxes_i = boxes[i]          
            classes_i = classes[i]       
            valid_mask = tf.reduce_any(boxes_i != 0, axis=-1)
            gt_boxes = tf.boolean_mask(boxes_i, valid_mask)
            gt_classes = tf.boolean_mask(classes_i, valid_mask)
            num_gt = tf.shape(gt_boxes)[0]
            if tf.equal(num_gt, 0):
                return tf.constant(0.0), tf.constant(0.0), tf.constant(0.0), tf.constant(0.0)

            iou_matrix = bbox_iou(gt_boxes, anchors_i)
            max_iou = tf.reduce_max(iou_matrix, axis=0)     
            best_gt_idx = tf.argmax(iou_matrix, axis=0)       

            pos_mask = max_iou >= pos_iou_thr
            neg_mask = max_iou < neg_iou_thr
            ignore_mask = ~(pos_mask | neg_mask)

            num_pos = tf.reduce_sum(tf.cast(pos_mask, tf.float32))

            if self.num_classes == 1:
                logits = cls_pred_i 
                gt_cls = tf.where(
                    tf.expand_dims(pos_mask, -1),
                    tf.ones_like(logits),
                    tf.zeros_like(logits)
                )
                cls_loss_all = tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_cls, logits=logits)
                cls_loss_all = tf.squeeze(cls_loss_all, axis=-1) 
            else:
                assigned_classes = tf.gather(gt_classes, best_gt_idx)  
                gt_cls = tf.one_hot(assigned_classes, depth=self.num_classes)  
                cls_loss_all = tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_cls, logits=cls_pred_i)
                cls_loss_all = tf.reduce_sum(cls_loss_all, axis=-1)  

            not_ignored_mask = ~ignore_mask
            not_ignored_float = tf.cast(not_ignored_mask, tf.float32)
            cls_loss = tf.reduce_sum(cls_loss_all * not_ignored_float)
          
            if tf.reduce_any(pos_mask):
                pos_indices = tf.where(pos_mask)[:, 0]
                pos_box_pred = tf.gather(box_pred_i, pos_indices)
                assigned_gt_boxes = tf.gather(gt_boxes, tf.gather(best_gt_idx, pos_indices))
                pos_anchors = tf.gather(anchors_i, pos_indices)

                gt_encoded = self._encode_boxes(assigned_gt_boxes, pos_anchors)

                diff = tf.abs(pos_box_pred - gt_encoded)
                box_loss = tf.where(
                    diff < 1.0,
                    0.5 * tf.square(diff),
                    diff - 0.5
                )
                box_loss = tf.reduce_sum(box_loss)
            else:
                box_loss = tf.constant(0.0, dtype=tf.float32)

            num_not_ignored = tf.reduce_sum(not_ignored_float)
            return cls_loss, box_loss, num_pos, num_not_ignored

        indices = tf.range(B, dtype=tf.int32)
        results = tf.map_fn(
            fn=process_single_element,
            elems=indices,
            fn_output_signature=(
                tf.float32,  # cls_loss
                tf.float32,  # box_loss
                tf.float32,  # num_pos
                tf.float32   # num_not_ignored
            )
        )
        cls_losses, box_losses, num_positives, num_not_ignored = results

        total_cls_loss = tf.reduce_sum(cls_losses)
        total_box_loss = tf.reduce_sum(box_losses)
        total_positives = tf.reduce_sum(num_positives)
        total_not_ignored = tf.reduce_sum(num_not_ignored)

        normalizer = tf.maximum(total_positives, 1.0)
        cls_loss = total_cls_loss / normalizer
        box_loss = total_box_loss / normalizer
        total_loss = cls_loss + box_loss

        #tf.print("total_pos:", total_positives, "total_not_ignored:", total_not_ignored)
        return total_loss, cls_loss, box_loss

    def train_step(self, data):
        images, targets = data

        with tf.GradientTape() as tape:
            cls_pred, box_pred = self(images, training=True)
            B = tf.shape(images)[0]
            anchors = self._get_anchors(B)
            total_loss, cls_loss, box_loss = self._compute_loss(
                cls_pred, box_pred, anchors, targets
            )
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {
            "loss": total_loss,
            "cls_loss": cls_loss,
            "box_loss": box_loss
        }

    def test_step(self, data):
        images, targets = data
        cls_pred, box_pred = self(images, training=False)
        B = tf.shape(images)[0]
        anchors = self._get_anchors(B)
        total_loss, cls_loss, box_loss = self._compute_loss(
            cls_pred, box_pred, anchors, targets
        )

        return {
            "loss": total_loss,
            "cls_loss": cls_loss,
            "box_loss": box_loss
        }

    def predict_boxes(self, images, score_threshold=0.5, max_detections=100):
        cls_pred, box_pred = self(images, training=False)

        if self.num_classes == 1:
            scores_all = tf.sigmoid(cls_pred)[:, :, 0] 
        else:
            scores_all = tf.reduce_max(tf.sigmoid(cls_pred), axis=-1) 
        B = tf.shape(images)[0]
        anchors = self._get_anchors(B)

        all_predictions = []

        for i in range(B):
            img_scores = scores_all[i]   
            img_boxes = box_pred[i]      
            img_anchors = anchors[i]     
            keep_mask = img_scores > score_threshold
            if not tf.reduce_any(keep_mask):
                all_predictions.append((tf.zeros([0, 4]), tf.zeros([0])))
                continue
            boxes_decoded = self._decode_boxes(
                tf.boolean_mask(img_boxes, keep_mask),
                tf.boolean_mask(img_anchors, keep_mask)
            )
            boxes_scores = tf.boolean_mask(img_scores, keep_mask)
            nms_indices = tf.image.non_max_suppression(
                boxes_decoded,
                boxes_scores,
                max_output_size=max_detections,
                iou_threshold=0.5,
                score_threshold=score_threshold
            )
            final_boxes = tf.gather(boxes_decoded, nms_indices)
            final_scores = tf.gather(boxes_scores, nms_indices)
            all_predictions.append((final_boxes, final_scores))
        return all_predictions
