
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


I am working on a machine learning project and created a virtual machine on google compute engine to train my models. I have access to this virtual machine directly from my local terminal through "gcloud compute ssh chillmate-vm2 --project chillmate-test1 ". I hae also access to the virtual machine from the SSH-in-browser terminal directly on compute engine. I started working on the virtual machine connecting frorm my local terminal and installed on it python 3.10.6. However, the python version on the virtual machine when connecting through SSH-in-browser looks different. For example, the python version is 3.8.10. Furthermore, a few packages I installed in the VM when connected from my terminal do not seem to be installed on the virtual machine when connected through the SSH-in-browser. Please explain why this difference and what to do to have the same changes and packages installed in the VM when connecting directly from y local terminal and when connecting from SSH-in-browser.
