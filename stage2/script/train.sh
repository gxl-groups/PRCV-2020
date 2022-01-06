###
 # @Author: JuncFang
 # @Date: 2021-04-07 10:12:33
 # @LastEditTime: 2021-04-07 11:17:30
 # @LastEditors: JuncFang
 # @Description: 
 # @FilePath: /stage2/script/train.sh
### 
CUDA_VISIBLE_DEVICES=0 python main_s2.py \
--pre_e ../stage1/checkpoint/E_epoch_89.pth \
--pre_g1 ../stage1/checkpoint/G1_epoch_89.pth \