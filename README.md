I generated a new image using stable diffusion and control net.

No , it's not just an API call.

I used Stable diffusion model and control net model with diffusers library on google colab to generate an image.

Let me explain the flow of the project :-

To run any AI model, we need a high end GPU. For this, I used Google Colab, which provides free access to a T4 GPU.

Now to run any ai model easily, we can use hugging face diffusers library. This library makes it easy to work with Stable Diffusion and ControlNet, offering pre built pipelines that streamline the process.
The best diffusion based model which we have currently is Stable diffusion model.

What does it mean by diffusion based model ?

In simple terms ,based on what I understand so far , diffusion based models work by first adding random noise (pixels) to an input image. This noise completely cover the image. The model then performs a reverse process, gradually removing the noise while generating a new image based on the given prompt and parameters.

This type of models can generate random images , the output image can be very different from the input image. It can be of any shape , size or pose.
To constrain the model and condition the model we will use ControlNet.

Simply, I used ControlNet to extract the edges of the input image, generating a control image. This control image helps the diffusion model maintain the shape of the output image. ControlNet uses the Canny edge detection algorithm to extract the edges.

ControlNet can also be used to capture poses, depth maps, and other structural features of an image. In this case, I used the sd-controlnet-canny model along with Stable Diffusion to ensure the final image maintain the desired structure.

On google colab:-
1. Installed the diffusers library.
2. Used this library to run Stable diffusion and control net library.
3. Used Diffusers pipelines to integrate Stable Diffusion (image generation model) and ControlNet (conditioning model). 

 Stable Diffusion generates images from text prompts. 
 ControlNet refines the image generation by adding structural constraints. 
 diffusers helps in merging both models into a single pipeline.

So, in Google Colab, you install diffusers, load Stable Diffusion + ControlNet, and then run image generation using Python code.

It's just a Proof of concept (POC) , just wanted to learn about genai , diffusion models , ai models.

I referred to a blog for all this.
