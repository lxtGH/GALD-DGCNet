## Some Advices on Training and Testing 
1. The output stride is 8 or 16 for keeping detailed information which lead to huge memory cost.

2. Non-local like networks usually use stride 8 for better performance.

3. Use OHEM loss for better performance on test set.

4. For Cityscape coarse dataset training, one can first train on fine-annotated data and then finetune on coarse dataset. 
Finally, re-finetune on the fine dataset.

5. For Mapillary dataset for the pretraining, it can boost preformance on CityScape or one can also use Maipillary as using the coarse data.

6. For testing, use the multi-scale cropping test with flip for the final test server submission.


# Notice
This code base is well orginized and its amis is to quick experiment and debug.