import tensorflow as tf
from random import randint

# Begin the session
session = tf.InteractiveSession()

# Sets the logging level for TensorFlow library.
tf.logging.set_verbosity(tf.logging.INFO)

# Import the image
image = tf.image.decode_png(tf.read_file("./data/train/0--Parade/0_Parade_marchingband_1_100.jpg"), channels=3)
session.run(image)

## Set the variable values here
# Offset variables values
offset_height= 200
offset_width = 200
# Target variables values
target_height = 200
target_width = 200

# Crop the image as per the parameters
cropped_image_tensor = tf.image.crop_to_bounding_box(image, offset_height, offset_width, target_height, target_width)

# Create a constant as filename
output_image = tf.image.encode_png(cropped_image_tensor)
file_name = tf.constant('./extract/train/0--Parade/0_Parade_marchingband_1_100.jpg/' + str(randint(0, 9)) +'1.jpg')
file = tf.write_file(file_name, output_image)

print(session.run(file))
print("Image Saved!")
session.close()
