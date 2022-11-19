#!/bin/bash

pip install tensorly
pip install timm

mkdir ./saved_models

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1UgP1zFavHGc7Jjre0YyXTxXCBYyJ-aqq' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1UgP1zFavHGc7Jjre0YyXTxXCBYyJ-aqq" -O ./saved_models/tkc_resnet18.pt && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Jhhrjlvd9byLIb7cUL2aU6d6i8kWOHbz' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Jhhrjlvd9byLIb7cUL2aU6d6i8kWOHbz" -O ./saved_models/tkc_resnet50.pt && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1_zYHW4xE7DUkV54pmPgX0wDYRIwxU9gI' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1_zYHW4xE7DUkV54pmPgX0wDYRIwxU9gI" -O ./saved_models/tkc_vgg16.pt && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1u8MXt1Z2XfTdiyABfx-nGwvsCzMxTaMo' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1u8MXt1Z2XfTdiyABfx-nGwvsCzMxTaMo" -O ./saved_models/tkc_densenet121.pt && rm -rf /tmp/cookies.txt

mkdir test_images
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1oRdLwEeY5YaW-szMgIykvBKxXzSeDLPG' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1oRdLwEeY5YaW-szMgIykvBKxXzSeDLPG" -O test_images/val.tar && rm -rf /tmp/cookies.txt

cd test_images
tar -xvf val.tar
