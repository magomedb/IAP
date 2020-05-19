# IAP
Create or use an existing UE4 project
Download the plugin from https://github.com/magomedb/IAP and export the IAP into your project’s plugins folder.
![Plugin folder](https://i.imgur.com/XIueRR7.png)

Now you need to decide if you are going to use GPU or CPU. If you don't plan on using GPU and would rather use CPU you can skip this paragraph. Note that GPU is a little faster then CPU, and a lot faster if you plan on using images. So if you decide to use GPU you need to first download CUDA 9.0, you can to that from here: https://developer.nvidia.com/cuda-90-download-archive
After you have downloaded CUDA 9.0 you need to download the corresponding CUDNN version. We suggest CUDNN v7.6.3 for CUDA 9.0, can be found here: https://developer.nvidia.com/rdp/cudnn-archive
Extract the CUDNN file and copy “cudnn64_7.dll” from “cuda\bin” to “C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin”, then copy “cudnn.h” from “cuda\include” to “C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include”, and at last copy “cudnn.lib” from “cuda\lib\x64” to “C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64”. Now you are done setting up CUDA and CUDNN for GPU support.

You also need pip for the plugins to be able automatically download its dependencies. Pip comes with any python version above 3.4. If you don't have pip here is an installation link for the python version we used with automatic pip installation: https://www.python.org/downloads/release/python-368/. Note that that the installation will ask you “Add Python (version) to PATH” you need accept that for pip to get setup. You also can’t use python above 3.7 because these versions do not support tensorflow 1.

Now you need to download the depending plugins. Go here: https://github.com/getnamo/tensorflow-ue4/releases, and download the version for UE4.23, and pick the GPU or CPU version depending on what you want to use.
![UE4TF](https://i.imgur.com/TkhfQ9n.png)

Now extract the plugins and copy them into the plugins folder in your project like so:
![in plugin folder](https://i.imgur.com/LBzEDvh.png)
Now you are ready to launch your project. Give some times the first time you launch the project as it will be setting up the plugins and that might take some time. After you have launched the project the plugin will install its dependencies, you can see when this is done in “Output Log”, when you see the print “Successfully installed absl-py-0.9.0” it will be done and the plugin is ready. 

# How to use
This will be a short tutorial or just explanation of how the most general functionality works and how-to setup a controller. For more information go to the examples in the plugin, these will show you how to use different versions with an environment. In addition if you want even more information on how it all works and more we will link you to our bachelor thesis, but we cannot do that before it is graded.

First you have to create a IAP controller class. You can do this by clicking "Add New" then "Blueprint Class"
![Create class](https://i.imgur.com/piIrUbC.png)

When you get inside the class you will be presented with a lot of options to your right.
![Options](https://i.imgur.com/qNGVmhH.png)

You can either choose input as handpicked values or images. For this example we will show handpicked values.
Now you need to setup these different inputs based on what you want. It is not that important to get stuff like learning rate correct right now, but set it to a reasonable value.

Now you need to setup your input, remember make sure you update amount precepts to correspond with how many input values you have. The way you setup this input is to add all the variables you want to the precept array in the precept struct, it will automatically be sent to the DQN.
![Input](https://i.imgur.com/it38KtZ.png)

Also note that we setup calculate reward also to be called each tick. We will get into this function later.

The next step is to setup your actions space. This the amount of actions needs to be set in amount actions.
![Actions](https://i.imgur.com/LypkyVu.png)

We also provide some actions like shoot closest target and move towards point that you can use.

The agent needs to be able to know what is good and bad. We do this by rewarding the agent in a calculatereward function. This function can be created in what ever way you want as long as you update the reward  variable. What we like to do is to have bools that the environment set then reward the agent based on these booleans and then reset them.
![Reward](https://i.imgur.com/7qsYhO8.png)

That is pretty much all you need to setup a controller. If you want to use some of the functionality  mentioned above you need to create a navmesh for the move towards point, go to the UE4 documentation if you don't know how that is done. Also, if you want to use shoot target functions you need to set weapon socket. This can be done like this:
![WeaponSocket](https://i.imgur.com/4RFVE6I.png)

We also have two addition programs, one is used to plot the files that is printed out by checking "Print RewardPr1k". The other program is used to calculate means and standard deviations for the use of z-score. This program uses the file created from the checkbox "Print Observations" and gives the calculations out in a format you can just copy paste into the means and standard deviations section so that the system will understand it.

If you want to know more specific use of this in an environment, you can look at any of the 4 examples we give. This can be located in IAP Content->AI->Examples.

![Examples](https://i.imgur.com/qWtLaE6.png)

# Packaging and running packaged build
When you are using the system in your own project you probably are going to want to package your project. Now to do this with the IAP and its dependencies you are going to have to be aware of some things. First of all you need to make sure your project has at least one CPP class. Yes it is weird, but it has something to do with the ue4-tensorflow plugin and the packaging of it. Meaning if you have a blueprint only project you need to add one C++ class. Do this by going to the bottom left corner and click “Add New”.

![CreateCpp](https://i.imgur.com/TkhfQ9n.png)

This should bring up a menu where you are going to click” New C++ Class”.

![CreateCpp2](https://i.imgur.com/MqVLZKM.png)

This will give you a progress bar that will have to finish before you can package your project. Note you only have to do this once per project.
Now that you have created a project with a C++ file you can package your project by clicking “File” in the top left, then “Package Project”, and pick your desired platform.

![Package](https://i.imgur.com/3H41Ppp.png)

Now that you have packaged the build through UE4 there are still a few things you need to do. First, go to the plugin folder of your packaged build and delete all the plugins inside, then copy over your plugins from your project to the packaged build. This is because some of the plugins we use don't always package correctly. Also make sure that the path is not too long because then the TensorFlow plugin will have some issues finding Tensorflow.

Now at last, for running the packaged build all you need is PIP installed and not to have the packaged build in a too long file path as mentioned above. If the computer running the packaged build does not have PIP we suggest downloading Python 3.6 because PIP is included in these versions. You can download Python here: https://www.python.org/downloads/release/python-368/. Note that this have to be python 3.4 to 3.7 as above does not support tensorflow 1.

# Videos
https://youtu.be/t9GQEJkPjQo
https://youtu.be/-LPg1Yo98zk
https://youtu.be/MUBJcMz4Ibw
https://youtu.be/9zMckgrrFpE
https://youtu.be/1BZ08_SqRIY
https://youtu.be/-179XsIUD3Y?t=17
https://youtu.be/skoQQijbipg
https://youtu.be/BGOQFHDh9no
https://youtu.be/rQqG8x8hTxw

# Credits
Getnamo:
Tensorflow UE4 and SocketIO Client UE4
https://github.com/getnamo/tensorflow-ue4
https://github.com/getnamo/socketio-client-ue4

20tab:
UnrealEnginePython
https://github.com/20tab/UnrealEnginePython

Dneproman:
Low Poly Universe(3d assets used in examples)
https://forums.unrealengine.com/community/community-content-tools-and-tutorials/90186-universe-content-1550-free-meshes

Pylessons:
Prioritized experience replay
https://pylessons.com/CartPole-PER/
