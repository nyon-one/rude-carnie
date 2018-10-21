import guess
import tensorflow as tf

GENDER_LIST = 'MF'

device_id = 'cpu:0'
model_type = 'inception'
model_dir = 21936

files = [r'C:\Users\dell\Desktop\animatetimes-695592997698711553-20160205_184608-img1.jpg']

config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(config=config) as sess:
    label_list = GENDER_LIST
    nlabels = len(label_list)

    print('Executing on %s' % device_id)
    model_fn = guess.select_model(model_type)

    with tf.device(device_id):
        images = tf.placeholder(tf.float32, [None, guess.RESIZE_FINAL, guess.RESIZE_FINAL, 3])
        logits = model_fn(nlabels, images, 1, False)
        init = tf.global_variables_initializer()
        
        checkpoint_path = '%s' % (model_dir)

        model_checkpoint_path, global_step = guess.get_checkpoint(checkpoint_path)

        saver = tf.train.Saver()
        saver.restore(sess, model_checkpoint_path)
                        
        softmax_output = tf.nn.softmax(logits)

        coder = guess.ImageCoder()

        writer = None
        output = None

        image_files = list(filter(lambda x: x is not None, [guess.resolve_file(f) for f in files]))
        # guess.classify_one_multi_crop(sess, label_list, softmax_output, coder, images, image_files, writer)
        guess.classify_many_single_crop(sess, label_list, softmax_output, coder, images, image_files, writer)




def main(argv=None):  # pylint: disable=unused-argument

    files = []
    
    if FLAGS.face_detection_model:
        print('Using face detector (%s) %s' % (FLAGS.face_detection_type, FLAGS.face_detection_model))
        face_detect = face_detection_model(FLAGS.face_detection_type, FLAGS.face_detection_model)
        face_files, rectangles = face_detect.run(FLAGS.filename)
        print(face_files)
        files += face_files

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:

        label_list = AGE_LIST if FLAGS.class_type == 'age' else GENDER_LIST
        nlabels = len(label_list)

        print('Executing on %s' % FLAGS.device_id)
        model_fn = select_model(FLAGS.model_type)

        with tf.device(FLAGS.device_id):
            
            images = tf.placeholder(tf.float32, [None, RESIZE_FINAL, RESIZE_FINAL, 3])
            logits = model_fn(nlabels, images, 1, False)
            init = tf.global_variables_initializer()
            
            requested_step = FLAGS.requested_step if FLAGS.requested_step else None
        
            checkpoint_path = '%s' % (FLAGS.model_dir)

            model_checkpoint_path, global_step = get_checkpoint(checkpoint_path, requested_step, FLAGS.checkpoint)
            
            saver = tf.train.Saver()
            saver.restore(sess, model_checkpoint_path)
                        
            softmax_output = tf.nn.softmax(logits)

            coder = ImageCoder()

            # Support a batch mode if no face detection model
            if len(files) == 0:
                if (os.path.isdir(FLAGS.filename)):
                    for relpath in os.listdir(FLAGS.filename):
                        abspath = os.path.join(FLAGS.filename, relpath)
                        
                        if os.path.isfile(abspath) and any([abspath.endswith('.' + ty) for ty in ('jpg', 'png', 'JPG', 'PNG', 'jpeg')]):
                            print(abspath)
                            files.append(abspath)
                else:
                    files.append(FLAGS.filename)
                    # If it happens to be a list file, read the list and clobber the files
                    if any([FLAGS.filename.endswith('.' + ty) for ty in ('csv', 'tsv', 'txt')]):
                        files = list_images(FLAGS.filename)
                
            writer = None
            output = None
            if FLAGS.target:
                print('Creating output file %s' % FLAGS.target)
                output = open(FLAGS.target, 'w')
                writer = csv.writer(output)
                writer.writerow(('file', 'label', 'score'))
            image_files = list(filter(lambda x: x is not None, [resolve_file(f) for f in files]))
            print(image_files)
            if FLAGS.single_look:
                classify_many_single_crop(sess, label_list, softmax_output, coder, images, image_files, writer)

            else:
                for image_file in image_files:
                    classify_one_multi_crop(sess, label_list, softmax_output, coder, images, image_file, writer)
            
            if output is not None:
                output.close()
