
1. Copy the project chillmate into the virtual machine if it's been updated. This has to be done from the parent folder of your project and from your local machine.

      gcloud compute scp --recurse chillmate-1372 $USER@chillmate-vm2:~/basic_model2 --project chillmate-test1

2. Connect to the virtual machine from terminal on local machine
      INSTANCE=chillmate-vm3
      gcloud compute ssh $chillmate-vm3

      You will be asked to enter a passphrase. I usually press enter and then enter again.

      You could also specify the project. But no always necessary.
      gcloud compute ssh chillmate-vm3 --project chillmate-test1

3. Once on the virtual machine, create folders. Do it only the first time.
      make reset_local_files -> to generate folders for the project

4. The first time you go into the virtual machine, install direnv
      sudo apt-get update
      sudo apt-get install direnv
5. rm .python-version -> to delete the call to the virtual env created on local machine. As part of the process of setting up a VM you should create a virtual environment.

6. Install package base_fruit_classifier. Different modules are modules
      pip install .

7.
Tensorboard access:

tensorboard --logdir=/home/andreslemus/project5/logs/adam/20240307-080015 --host=0.0.0.0 --port=6007


Install tree
sudo apt-get update
sudo apt-get install tree


Github process
- git status
- git add .
- git commit -m "added functions and structure for xception model"
- git checkout main
- git status
- git pull origin main
- git checkout xception_functions
- git merge main
- git checkout main
- git merge xception_functions
- git push origin main
- git checkout -b additional_models_vms
