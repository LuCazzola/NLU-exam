# NLU-exam
Exam repository of the Natural Language Understanding course, Artificial Intelligence Systems, Università di Trento (ITA)

## Weights download
If you come from GitHub, please before cloning the repository make sure to have **git lfs** on you system, otherwise loaded .pt will be unreadable
```
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install
```
Then proceed cloning the repository together with the model weights
```
git clone https://github.com/LuCazzola/NLU-exam.git
cd NLU-exam
git lfs pull
```
Then install requirements
```
pip install -r requirements.txt
```
