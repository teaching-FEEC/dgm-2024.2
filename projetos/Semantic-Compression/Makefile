data/coco: 
	mkdir -p data/coco
	wget  https://calvin-vision.net/wp-content/uploads/data/cocostuffdataset/cocostuff-10k-v1.1.zip
	unzip cocostuff-10k-v1.1.zip -d data/coco
	rm -rf cocostuff-10k-v1.1.zip
	mkdir -p data/coco/train/images
	mkdir -p data/coco/train/semantic
	mkdir -p data/coco/val/images
	mkdir -p data/coco/val/semantic
	python semantic-compression/dataset.py --dataset coco
	rm -rf data/coco/annotations	
	rm -rf data/coco/images
	rm -rf data/coco/imageLists
	mogrify -verbose -resize 384x384^ data/coco/*/images/*.jpg
	mogrify -verbose -interpolate Integer -filter point -resize 384x384^ data/coco/*/semantic/*.png

data/city:
	mkdir -p data/city
	python semantic-compression/dataset.py --dataset city
	unzip leftImg8bit_trainvaltest.zip -d data/city
	rm -rf leftImg8bit_trainvaltest.zip
	mkdir -p data/city/train/images
	mkdir -p data/city/val/images
	mv data/city/leftImg8bit/train/*/*.png data/city/train/images/
	mv data/city/leftImg8bit/test/*/*.png data/city/train/images/
	mv data/city/leftImg8bit/val/*/*.png data/city/val/images/
	rm -rf data/city/leftImg8bit
	mogrify -verbose -resize 384x384^ data/city/*/images/*.png
	unzip gtFine_trainvaltest.zip -d data/city
	rm -rf gtFine_trainvaltest.zip
	mkdir -p data/city/train/semantic
	mkdir -p data/city/val/semantic
	mv data/city/gtFine/train/*/*labelIds.png data/city/train/semantic/
	mv data/city/gtFine/test/*/*labelIds.png data/city/train/semantic/
	mv data/city/gtFine/val/*/*labelIds.png data/city/val/semantic/
	rm -rf data/city/gtFine
	mogrify -verbose -interpolate Integer -filter point -resize 384x384^ data/city/*/semantic/*.png

train: semantic-compression/config.py
	mkdir -p models
	python semantic-compression/modeling/train.py
