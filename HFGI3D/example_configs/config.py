## Pretrained models paths
e4e = './pretrained_models/e4e_ffhq_encode.pt'
stylegan2_ada_ffhq = '../ffhq512-128.pkl'
style_clip_pretrained_mappers = ''
ir_se50 = './pretrained_models/model_ir_se50.pth'
dlib = './pretrained_models/align.dat'

## Dirs for output files
checkpoints_dir = '../checkpoints'
# embedding_base_dir = '../embeddings'
embedding_base_dir = '../embeddings_z'
styleclip_output_dir = './StyleCLIP_results'
# experiments_output_dir = '../output'
experiments_output_dir = '../output_z'
multi_views_output_dir = '../multi_views_blending'
## Input info
### Input dir, where the images reside
input_data_path = '../../test_imgs/'
name = '00001'
### Inversion identifier, used to keeping track of the inversion results. Both the latent code and the generator
input_data_id = 'optimize_w+_500'#'barcelona'
## Keywords
pti_results_keyword = 'PTI'
e4e_results_keyword = 'e4e'
z_results_keyword = 'Z'
sg2_results_keyword = 'SG2'
sg2_plus_results_keyword = 'SG2_plus'
multi_id_model_type = 'multi_id'

##option
difference_map = False
sr_module = True
optimization = True

## Edit directions
# boundaries_dir = 'boundaries'
# boundaries_dir = 'boundaries_zp'
# boundaries_dir = 'boundaries_z'
boundaries_dir = 'boundaries_zpp'
interfacegan_age = 'editings/interfacegan_directions/age.pt'
interfacegan_smile = 'editings/interfacegan_directions/smile.pt'
interfacegan_rotation = 'editings/interfacegan_directions/rotation.pt'
ffhq_pca = 'editings/ganspace_pca/ffhq_pca.pt'
