#!/bin/bash
# Build Lambda deployment package

echo "🔨 Building Lambda deployment package..."

# Create temp directory
rm -rf lambda_package
mkdir -p lambda_package

# Copy Lambda handler
cp lambda_handler.py lambda_package/

# Copy KIS collector
cp ../lob_preprocessing/data/kis_lob_collector.py lambda_package/

# Install dependencies (INCLUDING requests!)
pip install -t lambda_package/ \
    requests \
    websocket-client \
    boto3 \
    pandas \
    pyarrow \
    python-dotenv \
    -q

# Create zip
cd lambda_package
zip -r ../lambda_kis.zip . -q
cd ..

# Cleanup
rm -rf lambda_package

echo "✅ Lambda package created: lambda_kis.zip"
ls -lh lambda_kis.zip
