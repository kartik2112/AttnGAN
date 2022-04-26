# Install reqmts
pip install -r requirements.txt

# Download coco.zip containing DAMSMencoders
gdown 1zIrXCE9F6yfbEJIbNP5-YrEe2pZcPSGJ -O DAMSMencoders
unzip DAMSMencoders/coco.zip -O DAMSMencoders/

# Download DAMSMencoders
gdown 1rSnbIGNDGZeHlsUlLdahj0RJ9oo6lgH9 -O data/
unzip data/coco.zip -O data

# Download images
gdown 1II0ijRjdPpSl0Nnz2ncktigGDujQ1nb_ -O data/coco/
unzip ./data/coco/images.zip -o data/coco/

# Download minicoco related files
gdown 10pyh2K8ybPdp0xBNTQ-4uSFSlBpVZQuL -O data/coco/
gdown 19TcpVKJHEUBBC7jCjqmtDmwi1CcftfQ5 -O data/coco/
gdown 1U3hD-pkA9L2sNjx5bYjXnwf0gvhXNHuU -O data/coco/
gdown 1zi-eGTN7Zqy-WyR02XdqbynUOE1CaDUg -O data/coco/