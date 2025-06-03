# 设置构建参数
ARG YOLOV_VERSION=latest

# 使用Ubuntu 24.04作为基础镜像
FROM ubuntu:24.04

# 设置环境变量
ENV LANG=C.UTF-8 \
    TZ=Asia/Shanghai \
    DEBIAN_FRONTEND=noninteractive \
    YOLOV_VERSION=${YOLOV_VERSION}

# 设置工作目录
WORKDIR /app

# 安装所有依赖并配置SSH（合并RUN命令以减少层级）
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    build-essential \
    ca-certificates \
    openssh-server \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/run/sshd \
    && sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# 安装Miniconda并配置环境（合并RUN命令以减少层级）
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh \
    && echo 'export PATH="/opt/conda/bin:${PATH}"' >> /root/.bashrc \
    && /opt/conda/bin/conda init bash

# 设置conda环境变量
ENV PATH="/opt/conda/bin:${PATH}"

# 复制项目文件并安装依赖（合并COPY和RUN命令以减少层级）
COPY requirements.txt .
RUN /opt/conda/bin/conda create -n yolo python=3.12 -y \
    && . /opt/conda/bin/activate yolo \
    && pip install -r requirements.txt

# 复制项目文件
COPY . .

# 配置SSH服务
EXPOSE 22

# 添加版本信息
LABEL org.opencontainers.image.version="${YOLOV_VERSION}" \
      org.opencontainers.image.description="YOLOv Docker Image" \
      org.opencontainers.image.source="https://github.com/relucy/yolo"

# 设置启动命令
CMD ["/bin/bash", "-c", "/etc/init.d/ssh start && tail -f /dev/null"] 