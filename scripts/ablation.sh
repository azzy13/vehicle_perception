pip install gdown

gdown 1iqhM-6V_r1FpOlOzrdP_Ejshgk0DxOob -O ablation.zip

directory="../model/ablation"

if [ -d "$directory" ]; then
    echo "Directory exists. Proceeding with extraction..."
else
    echo "Directory does not exist. Creating directory..."
    mkdir -p "$directory"
fi

unzip -d "$directory" ablation.zip

rm ablation.zip