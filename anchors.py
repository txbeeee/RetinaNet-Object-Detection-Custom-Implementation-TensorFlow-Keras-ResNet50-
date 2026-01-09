class AnchorBox:
    def __init__(self,
                 strides=(8, 16, 32),
                 sizes=(32, 64, 128),
                 ratios=(0.5, 1.0, 2.0),
                 scales=(1.0, 2 ** (1/3), 2 ** (2/3))):
        self.strides = list(strides)
        self.sizes = list(sizes)
        self.ratios = list(ratios)
        self.scales = list(scales)

        if len(self.strides) != len(self.sizes):
            raise ValueError("strides and sizes must have equal length.") #######

        self.num_levels = len(self.strides)
        self.num_anchors_per_location = len(self.ratios) * len(self.scales)

    def _generate_level_anchors(self, feat_h, feat_w, stride, size):
        shifts_x = tf.range(feat_w, dtype=tf.float32) * tf.cast(stride, tf.float32) + tf.cast(stride, tf.float32) / 2.0
        shifts_y = tf.range(feat_h, dtype=tf.float32) * tf.cast(stride, tf.float32) + tf.cast(stride, tf.float32) / 2.0
        shift_y, shift_x = tf.meshgrid(shifts_y, shifts_x)
        shift_x = tf.reshape(shift_x, [-1])
        shift_y = tf.reshape(shift_y, [-1])

        base_area = tf.cast(size * size, tf.float32)
        ws, hs = [], []
        for r in self.ratios:
            r = tf.cast(r, tf.float32)
            for s in self.scales:
                s = tf.cast(s, tf.float32)
                area = base_area * (s ** 2)
                w = tf.sqrt(area / r)
                h = w * r
                ws.append(w)
                hs.append(h)

        ws = tf.reshape(tf.convert_to_tensor(ws, tf.float32), [1, self.num_anchors_per_location])
        hs = tf.reshape(tf.convert_to_tensor(hs, tf.float32), [1, self.num_anchors_per_location])

        cx = tf.reshape(shift_x, [-1, 1])
        cy = tf.reshape(shift_y, [-1, 1])
        x1 = cx - ws / 2.0
        y1 = cy - hs / 2.0
        x2 = cx + ws / 2.0
        y2 = cy + hs / 2.0
        anchors = tf.stack([x1, y1, x2, y2], axis=2)
        return tf.reshape(anchors, [-1, 4])

    def from_feature_maps(self, feature_shapes):
        anchors_all = []
        for i, (h, w) in enumerate(feature_shapes):
            stride_h = IMAGE_SIZE / float(h)
            stride_w = IMAGE_SIZE / float(w)
            stride = float((stride_h + stride_w) / 2.0)

            size = self.sizes[i] if i < len(self.sizes) else self.sizes[-1]
            anchors_all.append(self._generate_level_anchors(h, w, stride, size))
        return tf.concat(anchors_all, axis=0)


def bbox_iou(boxes1, boxes2):
    boxes1 = tf.expand_dims(boxes1, 1)  # [N 1 4]
    boxes2 = tf.expand_dims(boxes2, 0)  # [1 M 4]

    x1 = tf.maximum(boxes1[..., 0], boxes2[..., 0])
    y1 = tf.maximum(boxes1[..., 1], boxes2[..., 1])
    x2 = tf.minimum(boxes1[..., 2], boxes2[..., 2])
    y2 = tf.minimum(boxes1[..., 3], boxes2[..., 3])

    intersection = tf.maximum(0.0, x2 - x1) * tf.maximum(0.0, y2 - y1)

    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    union = area1 + area2 - intersection

    return intersection / (union + 1e-8)
