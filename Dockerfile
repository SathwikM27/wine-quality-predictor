# Use OpenJDK 8 as the base image
FROM openjdk:8-jre-slim

# Set environment variables
ENV SPARK_VERSION=3.5.2
ENV HADOOP_VERSION=3
ENV SPARK_HOME=/opt/spark
ENV PATH="$SPARK_HOME/bin:$PATH"

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    wget \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Install Spark
RUN wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz -P /tmp \
    && tar -xzf /tmp/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz -C /opt \
    && ln -s /opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark \
    && rm /tmp/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz

# Add Spark default configurations
RUN echo "spark.hadoop.io.compression.codecs=org.apache.hadoop.io.compress.DefaultCodec" > /opt/spark/conf/spark-defaults.conf \
    && echo "spark.executor.extraClassPath /dev/null" >> /opt/spark/conf/spark-defaults.conf \
    && echo "spark.driver.extraClassPath /dev/null" >> /opt/spark/conf/spark-defaults.conf

# Install Python dependencies
COPY requirements.txt /
RUN pip3 install --no-cache-dir -r /requirements.txt

# Set working directory
WORKDIR /app

# Copy the application code
COPY predict.py /app/

# Set the default command
CMD ["spark-submit", "predict.py"]
