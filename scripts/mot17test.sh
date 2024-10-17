pip install gdown

gdown 1XwfUuCBF4IgWBWK2H7oOhQgEj9Mrb3rz mot17test.zip

directory="../model/mot17test"

if [ -d "$directory" ]; then
    echo "Directory exists. Proceeding with extraction..."
else
    echo "Directory does not exist. Creating directory..."
    mkdir -p "$directory"
fi

unzip -d "$directory" mot17test.zip

rm mot17test.zip