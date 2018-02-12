# automatically remove the generated images with this script

#!/bin/bash
cd ~/deep-margins/tutorial/training_data/cats
echo *_*.jpg | xargs rm
cd ../dogs
echo *_*.jpg | xargs rm
echo "Generated images removed."
