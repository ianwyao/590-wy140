#USEFUL GIT COMMANDS 
#REFERENCE: http://guides.beanstalkapp.com/version-control/common-git-commands.html

# 1) INIT: "git init" is run from the command line and initializes a new repo (directory). 

git init 590-DEMOS-W2 # --> Initialized empty Git repository in /home/jfh/590-DEMOS-W2/.git/
 
# 2) ADD: "Git add" Adds files in the to the staging area for Git

# To stage a specific file:
git add test.dat 

# To add all files not staged:
$ git add .

# To stage an entire directory:
$ git add css

# 3) COMMIT: "git commit" Record the changes made to the files to a local repository. (each commit has a unique ID)
#         -should also include a message with each commit 

git commit -m "Commit message in quotes"

# 4) git status  # This command returns the current state of the repository.

# 5) git config  # git config assigns git settings. Two important settings are user user.name and user.email. 
                 # These values set what email address and name commits will be from on a local computer.
                 
git config --global user.email "you@example.com"
git config --global user.name "Your Name"


# 6) git clone  --> create a local working copy of an existing remote repository, use git clone to copy and download the repository to a computer. 

git clone <remote_URL>


# 7) git pull -->  get the latest version of a repository run git pull.

git pull <branch_name> <remote_URL/remote_name>


# 8) git push --> Sends local commits to the remote repository. git push requires two parameters: the remote repository and the branch that the push is for.
      #Usage:  git push <remote_URL/remote_name> <branch>

# Push all local branches to remote repository
git push —all




# 6) git branch To determine what branch the local repository is on, add a new branch, or delete a branch.

# Create a new branch
$ git branch <branch_name>

# List all remote or local branches
$ git branch -a

# Delete a branch
$ git branch -d <branch_name>


#git add ./; git commit -m "all"; git push

# git add SciPyRegression.py;  git commit -m "add SciPyRegression.py";   git push
# [main 38a07d2] add SciPyRegression.py
#  1 file changed, 54 insertions(+)
#  create mode 100644 SciPyRegression.py
# Username for 'https://github.com': GUID@georgetown.edu
# Password for 'https://GUID@georgetown.edu@github.com': 
# Counting objects: 3, done.
# Compressing objects: 100% (3/3), done.
# Writing objects: 100% (3/3), 954 bytes | 954.00 KiB/s, done.
# Total 3 (delta 1), reused 0 (delta 0)
# remote: Resolving deltas: 100% (1/1), completed with 1 local object.
# To https://github.com/jh2343/590-DEMOS-W1.git
#    8f1de7f..38a07d2  main -> main

#I HAVE WORKED THROUGH THIS EXAMPLE AND UNDERSTAND IT COMPLETELY

