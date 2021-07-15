import cv2 as cv
import utils
import os
import glob
import matplotlib.pyplot as plt


def main():
  
    #====  experiments ===========================================================================
    
    folders = os.listdir(DATASET_PATH)
    
    #onomata arxikon eikonon 
    
    pos_images_names = [glob.glob(os.path.join(DATASET_PATH, folder, "*post-contrast.tif_removed_background.png"))[0] for folder in folders]

    flair_images_names = [glob.glob(os.path.join(DATASET_PATH, folder, "*flair.tif_removed_background.png"))[0] for folder in folders]
    
    pre_images_names = [glob.glob(os.path.join(DATASET_PATH, folder, "*pre-contrast.tif_removed_background.png"))[0] for folder in folders]
    
    all_images = [pos_images_names, flair_images_names,pre_images_names]
       
    channel = 0
    
    #epilogi kanalioy
    
    images_names = all_images[channel]
   
    images = [ cv.imread(x, 0)  for x in images_names ]
    
    
    
    utils.plot_histograms(images,images_names,text='original histograms ')
    
    print("original histogram distance :")
    
    print(utils.histograms_compare(images,images_names))
    
    
#CLAHE =========================================================================================================
    
    final_images =utils.histogram_equalization_CLAHE(images,tile_grid_size=(24,24), clip_limit=10,images_name=images_names)
    
    
#=========================================================================================================================
#histogram matching==========================================================================================
 
#    ref_image_name = ['data/dataset/TCGA_FG_5964_20010511/TCGA_FG_5964_20010511_5_post-contrast.tif_removed_background.png',
#                      'data/dataset/TCGA_FG_6692_20020606/TCGA_FG_6692_20020606_4_flair.tif_removed_background.png',
#                      'data/dataset/TCGA_FG_A4MT_20020212/TCGA_FG_A4MT_20020212_5_pre-contrast.tif_removed_background.png']
#
#    ref_image = cv.imread(ref_image_name[channel],0)
#    
#    final_images = utils.histogram_matching(images,ref_image,images_name=images_names)
   
#=========================================================================================================================

#pipeline  average_hist -> exact_hist matching ====================================================================
    
    average_hist_img  = utils.avr_image(images)
    
    final_images = utils.exact_histogram_matching(images,average_hist_img)
#=========================================================================================================================
  
#  pipeline  histogram matching reference image -> clahe 10 ============================================================
 
#    ref_image_name = ['data/dataset/TCGA_FG_5964_20010511/TCGA_FG_5964_20010511_5_post-contrast.tif_removed_background.png',
#                      'data/dataset/TCGA_FG_6692_20020606/TCGA_FG_6692_20020606_4_flair.tif_removed_background.png',
#                      'data/dataset/TCGA_FG_A4MT_20020212/TCGA_FG_A4MT_20020212_5_pre-contrast.tif_removed_background.png']
#
#    ref_image = cv.imread(ref_image_name[channel],0)
#    
#    hist_images = utils.histogram_matching(images,ref_image,images_name=images_names)
#    
#    final_images =utils.histogram_equalization_CLAHE(hist_images,tile_grid_size=(24,24), clip_limit=10,images_name=images_names)
#=========================================================================================================================    
 
 
    print("results ===========================")
    
    print("mean histogram  distance :")
    
    print(utils.histograms_compare(final_images,images_names))
    
    print('SSIM >0.5:')
    
    print(utils.ssim_compare(final_images,images,images_names))
    
    print('MSE <1000:')
    
    print(utils.mse_compare(final_images,images,images_names))
#
#   save image
#    text = 'wh'
#    
#    for i in range(0,len(final_images)):
#        cv.imwrite('images/'+pos_images_names[i][-64:-27]+text+'.png',final_images[i])
    

if __name__ == '__main__':
    PARAMETERS_PATH = os.path.join('radiomics_modules', 'Params.yaml')

    DATASET_PATH = os.path.join('data', 'dataset')
    FEATURES_OUTPUT_PATH = os.path.join('data', 'pyradiomics_extracted_features.csv')

    NEW_DATASET_OUTPUT_PATH = os.path.join('data', 'new_dataset')
    NEW_FEATURES_OUTPUT_PATH = os.path.join('data', 'new_pyradiomics_extracted_features.csv')

    main()

