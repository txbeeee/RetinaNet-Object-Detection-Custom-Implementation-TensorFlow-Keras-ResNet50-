def load_coco_records(coco_json_path: Path, images_dir: Path):
    coco = json.load(open(coco_json_path, "r", encoding="utf-8"))
    id2img = {img["id"]: img for img in coco.get("images", [])}
    anns_by_img = {}
    for ann in coco.get("annotations", []):
        anns_by_img.setdefault(ann["image_id"], []).append(ann)
    records = []
    for img_id, info in id2img.items():
        p = images_dir / info["file_name"]
        if not p.exists():
            continue
        boxes, classes = [], []
        for a in anns_by_img.get(img_id, []):
            bbox = a.get("bbox", None)
            if bbox is None or len(bbox) < 4:
                continue
            x, y, w, h = map(float, bbox[:4])
            x1, y1, x2, y2 = x, y, x + w, y + h
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            classes.append(int(a.get("category_id", 0)))
        if len(boxes) == 0:
            continue

        records.append({
            "image_path": str(p),
            "width": int(info.get("width", 0) or 0),
            "height": int(info.get("height", 0) or 0),
            "boxes": np.array(boxes, dtype=np.float32),
            "classes": np.array(classes, dtype=np.int32),
        })
    return records
  
def resnet_preprocess(x):
    x = tf.cast(x, tf.float32)
    # RGB2BGR
    x = x[..., ::-1]
    mean = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32)
    x = x - mean
    return x

def resize_to_fixed(image, boxes, target_size=IMAGE_SIZE):
    h = tf.shape(image)[0]
    w = tf.shape(image)[1]
    image = tf.image.resize(image, [target_size, target_size], antialias=True)

    scale_x = tf.cast(target_size, tf.float32) / tf.cast(w, tf.float32)
    scale_y = tf.cast(target_size, tf.float32) / tf.cast(h, tf.float32)

    boxes = tf.stack([
        boxes[:, 0] * scale_x,
        boxes[:, 1] * scale_y,
        boxes[:, 2] * scale_x,
        boxes[:, 3] * scale_y
    ], axis=-1)

    boxes = tf.clip_by_value(boxes, 0.0, tf.cast(target_size, tf.float32))
    return image, boxes, (scale_x, scale_y)

def make_dataset(records, batch_size=BATCH_SIZE, shuffle=True, training=True):
    output_signature = {
        "image_path": tf.TensorSpec(shape=(), dtype=tf.string),
        "boxes": tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
        "classes": tf.TensorSpec(shape=(None,), dtype=tf.int32),
    }

    def gen():
        for r in records:
            yield {
                "image_path": r["image_path"],
                "boxes": np.asarray(r["boxes"], dtype=np.float32),
                "classes": np.asarray(r["classes"], dtype=np.int32),
            }

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    if shuffle:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)

    def _load(path, boxes, classes):
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]

        if training:
            flip = tf.less(tf.random.uniform([]), 0.5)
            img = tf.cond(flip, lambda: tf.image.flip_left_right(img), lambda: img)
            w = tf.cast(tf.shape(img)[1], tf.float32)
            boxes = tf.cond(
                flip,
                lambda: tf.stack([
                    w - boxes[:, 2], boxes[:, 1], w - boxes[:, 0], boxes[:, 3]
                ], axis=-1),
                lambda: boxes
            )
            img = tf.image.random_brightness(img, max_delta=0.1)
            img = tf.image.random_contrast(img, lower=0.9, upper=1.1)

        img, boxes, _ = resize_to_fixed(img, boxes, target_size=IMAGE_SIZE)
        img_pp = resnet_preprocess(img * 255.0)
        classes = tf.cast(classes, tf.int32)
        return img_pp, {"boxes": boxes, "classes": classes}

    ds = ds.map(lambda r: _load(r["image_path"], r["boxes"], r["classes"]),
                num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.padded_batch(
        batch_size,
        padded_shapes=(
            [IMAGE_SIZE, IMAGE_SIZE, 3],
            {
                "boxes": [None, 4],
                "classes": [None]
            }
        ),
        padding_values=(
            0.0,
            {
                "boxes": tf.constant(0.0, tf.float32),
                "classes": tf.constant(0, tf.int32)
            }
        ),
        drop_remainder=False
    ).prefetch(tf.data.AUTOTUNE)

    return ds
