curl http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip -o "./data/preprocess/#1"
curl http://images.cocodataset.org/zips/train2014.zip -o "./data/preprocess/#1"
curl http://images.cocodataset.org/zips/val2014.zip -o "./data/preprocess/#1"

unzip ./data/preprocess/captions_train-val2014.zip -d ./data/preprocess
rm ./data/preprocess/captions_train-val2014.zip
unzip ./data/preprocess/train2014.zip -d ./data/preprocess
rm ./data/preprocess/train2014.zip 
unzip ./data/val2014.zip -d ./data/preprocess
rm ./data/preprocess/val2014.zip 
