gtidevices=$(ls -1 /dev/sg[0-9]*|sort -V|tail -1|sed 's/^/--device=/')
#docker run --gpus=1 --rm -it -v ${PWD}/:/workspace/ -v /dev/sg2:/dev/sg2 -p 10022:22 -p 8000:8000 ${gtidevices}  -u root:root pytorch/pytorch bash 
NV_GPU=2 nvidia-docker run --rm -it -v ${PWD}/:/workspace/ -v /dev/sg2:/dev/sg2 -p 10022:22 -p 8000:8000 ${gtidevices}  pytorch/pytorch bash
