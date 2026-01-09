def main():
    # DATA_tr
    train_records = load_coco_records(TRAIN_JSON, TRAIN_IMAGES)
    if len(train_records) == 0:
        raise SystemExit("No training records found.")
    print(f"Loaded {len(train_records)} training records")
  
    all_cats = sorted({int(c) for r in train_records for c in r["classes"]})
    #reindexing
    cat_map = {old: i for i, old in enumerate(all_cats)}
    for r in train_records:
        r["classes"] = np.array([cat_map[int(c)] for c in r["classes"]], dtype=np.int32)
    global NUM_CLASSES
    NUM_CLASSES = len(all_cats)
    print(f"NUM_CLASSES: {NUM_CLASSES}, category map: {cat_map}")

    # Data_val
    val_records = []
    if VAL_JSON.exists():
        val_records = load_coco_records(VAL_JSON, VAL_IMAGES)
        for r in val_records:
            r["classes"] = np.array([cat_map[int(c)] for c in r["classes"]], dtype=np.int32)
        print(f"Loaded {len(val_records)} validation records")

    train_ds = make_dataset(train_records, batch_size=BATCH_SIZE, shuffle=True, training=True)
    val_ds = make_dataset(val_records, batch_size=BATCH_SIZE, shuffle=False, training=False) if val_records else None

    # Test pipeline
    try:
        sample_images, sample_targets = next(iter(train_ds))
        print(f"\nSample batch - Images shape: {sample_images.shape}")
        print(f"Sample batch - Boxes shape: {sample_targets['boxes'].shape}")
        print(f"Sample batch - Classes shape: {sample_targets['classes'].shape}")
    except Exception as e:
        print(f"Error in data pipeline: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"\nCreating model with {NUM_CLASSES} classes...")
    model = SimpleRetinaNet(num_classes=NUM_CLASSES)
    dummy_input = tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])
    _ = model(dummy_input, training=False)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LR)
    )

    print(f"\nModel summary:")
    model.summary()
    if sample_images is not None:
        anchors_one = model._get_anchors(1)[0].numpy()  
        dummy = tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3], dtype=tf.float32)
        features = model.backbone(dummy, training=False)
        print("Feature map shapes from backbone/FPN:")
        for i, f in enumerate(features):
            print(f"  level {i}: shape = {f.shape}")  

        print("AnchorBox.strides:", model.anchor_box.strides)
        print("AnchorBox.sizes:", model.anchor_box.sizes)
        print("AnchorBox.num_levels:", model.anchor_box.num_levels)
        print("___________________________________________________________")
        print("\n--- Anchor diagnostics ---")
        print("image shape:", sample_images.shape)   
        print("anchors shape:", anchors_one.shape)  
        print("anchors x min/max:", anchors_one[:,0].min(), anchors_one[:,2].max())
        print("anchors y min/max:", anchors_one[:,1].min(), anchors_one[:,3].max())

        centers = np.stack([(anchors_one[:,0] + anchors_one[:,2]) / 2.0,
                            (anchors_one[:,1] + anchors_one[:,3]) / 2.0], axis=-1)
        print("centers x range:", centers[:,0].min(), centers[:,0].max())
        print("centers y range:", centers[:,1].min(), centers[:,1].max())

        gt_boxes = sample_targets["boxes"][0].numpy()
        valid_mask = (gt_boxes.sum(axis=-1) > 0)
        gt_boxes = gt_boxes[valid_mask]
        print("num GT boxes in sample:", len(gt_boxes))

        if len(gt_boxes) > 0:
            iou_matrix = bbox_iou(tf.constant(gt_boxes), tf.constant(anchors_one)).numpy() 
            max_iou = iou_matrix.max(axis=0)
            print("max_iou stats: min, 25%, median, mean, 75%, max ->",
                  np.min(max_iou),
                  np.percentile(max_iou, 25),
                  np.median(max_iou),
                  np.mean(max_iou),
                  np.percentile(max_iou, 75),
                  np.max(max_iou))
        else:
            print("No GT boxes found in this sample (after padding filter).")
    else:
        print("No sample batch available for diagnostics.")

    ckpt_dir = OUTPUT_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_dir / "model_epoch_{epoch:02d}.h5"),
            save_weights_only=True,
            save_best_only=True,
            monitor='val_loss' if val_ds else 'loss',
            mode='min'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss' if val_ds else 'loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss' if val_ds else 'loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            str(OUTPUT_DIR / "training_log.csv"),
            append=True
        )
    ]
    steps_per_epoch = max(1, len(train_records) // BATCH_SIZE)
    validation_steps = max(1, len(val_records) // BATCH_SIZE) if val_ds else None
    
    sample_images, sample_targets = next(iter(train_ds))

    visualize_batch_anchors(model, sample_images, sample_targets)

    print(f"\nTraining configuration:")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Learning rate: {LR}")
    print(f"  Training samples: {len(train_records)}")
    print(f"  Validation samples: {len(val_records) if val_ds else 0}")

    history = model.fit(
        train_ds.repeat(),
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds if val_ds else None,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )

    final_model_path = OUTPUT_DIR / "retinanet_24.12.h5"
    model.save_weights(str(final_model_path))
    print(f"\nModel saved to: {final_model_path}")

    test_predictions = model.predict_boxes(sample_images[:1], score_threshold=0.3)
    boxes, scores = test_predictions[0]

    print(f"Predicted {len(boxes)} boxes")
    if len(boxes) > 0:
        print(f"Sample box: {boxes[0].numpy()}, score: {scores[0].numpy():.3f}")

    return history
