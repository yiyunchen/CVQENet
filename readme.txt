Test Code for CVQENet: CVQENet: Deformable Convolution-based Compressed Video Quality Enhancement Network, which took the 8th place in Track1 of the NTIRE 2021 Challenge on Quality Enhancement of Compressed Video.

Environment:
pytorch:1.2

Prerequest: 
1. cd code/ops/dcn/
2. bash build.sh
3. python simple_check.py

Test:
1. cd code
2. option setting
   - pretrain: the path of the pretrained model, e.g. ../pretrainModel
   - nb1 : the number of reblock in FEM, our pretrained model is nb1 = 10
   - nb2 : the number of reblock in FQEM, our pretrained model is nb2 = 20
   - nf : the number of channel, our pretrained model is nf = 64
   - test_dir: the path of testing video images
   - image_out: the path to save the output image
3. run code, python test.py

Acknowledgement:
our code is based on : https://github.com/RyanXingQL/STDF-PyTorch and https://github.com/xinntao/EDVR
