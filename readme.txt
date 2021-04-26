Test Code for CVQENet
prerequest: 
1. cd code/ops/dcn/
2. bash build.sh
3. python simple_check.py

test:
1. cd code
2. option setting
   - pretrain: the path of the pretrained model, e.g. ../pretrainModel
   - nb1 : the number of reblock in FEM, our pretrained model is nb1 = 10
   - nb2 : the number of reblock in FQEM, our pretrained model is nb2 = 20
   - nf : the number of channel, our pretrained model is nf = 64
   - test_dir: the path of testing video images
   - image_out: the path to save the output image
3. run code, python test.py
