def visualize_batch_anchors(model, images, targets, score_threshold=0.0, max_anchors=500):
    cls_pred, box_pred = model(images, training=False)
    B = images.shape[0]
    anchors = model._get_anchors(B)

    for i in range(B):
        img = images[i].numpy().astype("float32")
        img = img[..., ::-1] / 255.0  # BGR2RGB
        gt_boxes = targets["boxes"][i].numpy()
        valid_mask = (gt_boxes.sum(axis=-1) > 0)
        gt_boxes = gt_boxes[valid_mask]
        iou_matrix = bbox_iou(gt_boxes, anchors[i])
        max_iou = tf.reduce_max(iou_matrix, axis=0).numpy()
        pos_mask = max_iou >= 0.5
        neg_mask = max_iou < 0.4
        ignore_mask = ~(pos_mask | neg_mask)
      
        A = anchors[i].numpy()
        if A.shape[0] > max_anchors:
            idx = np.random.choice(A.shape[0], max_anchors, replace=False)
            A = A[idx]
            pos_mask = pos_mask[idx]
            neg_mask = neg_mask[idx]
            ignore_mask = ignore_mask[idx]

        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(img)
        ax.set_title(f"Image {i}: GT (green), Pos anchors (red), Neg anchors (blue), Ignore (yellow)")

        for (x1, y1, x2, y2) in gt_boxes:
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                     linewidth=2, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)

        for j, (x1, y1, x2, y2) in enumerate(A):
            color = 'yellow'
            if pos_mask[j]:
                color = 'red'
            elif neg_mask[j]:
                color = 'blue'

            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                     linewidth=0.5, edgecolor=color, facecolor='none', alpha=0.5)
            ax.add_patch(rect)

        plt.show()
