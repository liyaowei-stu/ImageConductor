CUDA_VISIBLE_DEVICES=0
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python -m main.inference \
--config configs/prompt/trajs/object.yaml \



CUDA_VISIBLE_DEVICES=0
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python -m main.inference \
--config configs/prompt/camera/camera.yaml \



if [ $? != 0 ]; then
   echo "Fail! Exit with 1"
   exit 1
else
   echo "Success! Exit with 0"
   exit 0
fi
