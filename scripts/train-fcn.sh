#!/bin/bash

# 使用示例: bash scripts/train.sh logger=wandb devices="[0,1,2,3]" max_epochs=50
# 默认参数
LOGGER="wandb"
DEVICES="[0]"
MAX_EPOCHS=150

# 解析命令行传递的参数
for arg in "$@"
do
    case $arg in
        logger=*)
            LOGGER="${arg#*=}"
            ;;
        devices=*)
            DEVICES="${arg#*=}"
            ;;
        max_epochs=*)
            MAX_EPOCHS="${arg#*=}"
            ;;
        *)
            # 不支持的参数
            echo "不支持的参数: $arg"
            exit 1
            ;;
    esac
done

# 训练的模型列表
experiments=("fcn-densenet161" "fcn-dpn131" "fcn-inceptionv4" "fcn-mit_b5" "fcn-mobileone_s4" "fcn-resnet152" "fcn-resnext101_32x48d" "fcn-se_resnext101_32x4d" "fcn-timm-efficientnet-l2" "fcn-timm-gernet_l" "fcn-timm-mobilenetv3_large_100" "fcn-timm-regnetx_320" "fcn-timm-res2net101_26w_4s" "fcn-timm-resnest101e")

# 遍历每个实验配置并运行训练
for experiment in "${experiments[@]}"
do
  echo "开始训练: $experiment 使用 $LOGGER 在设备 $DEVICES"
  python src/train.py experiment=$experiment trainer.devices=$DEVICES logger=$LOGGER trainer.max_epochs=$MAX_EPOCHS
done