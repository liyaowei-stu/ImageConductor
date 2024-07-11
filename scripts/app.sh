CUDA_VISIBLE_DEVICES=0
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ImageConductor_app.py



if [ $? != 0 ]; then
   echo "Fail! Exit with 1"
   exit 1
else
   echo "Success! Exit with 0"
   exit 0
fi
