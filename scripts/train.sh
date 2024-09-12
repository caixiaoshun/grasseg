#!/bin/bash

# 默认参数
LOGGER="wandb"
DEVICES="[1]"

# 解析命令行传递的参数
for arg in "$@"
do
    case $arg in
        logger=*)
        LOGGER="${arg#*=}"
        shift
        ;;
        devices=*)
        DEVICES="${arg#*=}"
        shift
        ;;
        *)
        # 其它未处理的参数
        ;;
    esac
done

# 训练的模型列表
experiments=("farseg_resnet18" "farseg_resnet34" "farseg_resnet50" "farseg_resnet101" "fcn" "unetmobv2")

# 遍历每个实验配置并运行训练
for experiment in "${experiments[@]}"
do
  echo "开始训练: $experiment 使用 $LOGGER 在设备 $DEVICES"
  python src/train.py experiment=$experiment trainer.devices=$DEVICES logger=$LOGGER
done
