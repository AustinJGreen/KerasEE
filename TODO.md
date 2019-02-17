Some sort of assistant instead of inpainting
One model needs to output a prediction of an added block or blocks?
One model predicts how good that block is

32x32 Area, one model outputs prediction score for good each block is pixelwise
other model is purely given 32x32 input like auto-encoder and tasked to improve area

Judge is trained upon whether or not original comes from generated, 100% or 0% sure
Helper is trained upon

Create model that takes in 64x64 input and outputs likelihood of 1 block in the center of the input

Train multiple classifiers for each type
- Should help my ability to build a better data set
- Should help classifier's performance
- Flexible for more classifers in the future (animator, stairs, boss, etc...)

- Add more data, scrape by authors instead of by likes
    - Can scrape authors from current worlds folder

- Model that turns image into handmade art
    - Actually might be pretty easy, just train network to produce blocks from minimap
        - Need to only allow blocks with minimap values