IMPL_MODEL_NAMES = {
    'torchvision': [
        'alexnet',
        'resnet18',
        'resnet34',
        'resnet50',
        'resnet101',
        'resnet152',
        'resnext50_32x4d',
        'resnext101_32x8d',
        'wide_resnet50_2',
        'wide_resnet101_2',
        'vgg11',
        'vgg11_bn',
        'vgg13',
        'vgg13_bn',
        'vgg16',
        'vgg16_bn',
        'vgg19_bn',
        'vgg19',
        'squeezenet1_0',
        'squeezenet1_1',
        'inception_v3',
        'densenet121',
        'densenet169',
        'densenet201',
        'densenet161',
        'googlenet',
        'mobilenet_v2',
        'mobilenet_v3_large',
        'mobilenet_v3_small',
        'mnasnet0_5',
        'mnasnet0_75',
        'mnasnet1_0',
        'mnasnet1_3',
        'shufflenet_v2_x0_5',
        'shufflenet_v2_x1_0',
        'shufflenet_v2_x1_5',
        'shufflenet_v2_x2_0'
    ],
    'pytorchcv': [
        'alexnet',
        'alexnetb',
        'zfnet',
        'zfnetb',
        'vgg11',
        'vgg13',
        'vgg16',
        'vgg19',
        'bn_vgg11',
        'bn_vgg13',
        'bn_vgg16',
        'bn_vgg19',
        'bn_vgg11b',
        'bn_vgg13b',
        'bn_vgg16b',
        'bn_vgg19b',
        'bninception',
        'resnet10',
        'resnet12',
        'resnet14',
        'resnetbc14b',
        'resnet16',
        'resnet18_wd4',
        'resnet18_wd2',
        'resnet18_w3d4',
        'resnet18',
        'resnet26',
        'resnetbc26b',
        'resnet34',
        'resnetbc38b',
        'resnet50',
        'resnet50b',
        'resnet101',
        'resnet101b',
        'resnet152',
        'resnet152b',
        'resnet200',
        'resnet200b',
        'preresnet10',
        'preresnet12',
        'preresnet14',
        'preresnetbc14b',
        'preresnet16',
        'preresnet18_wd4',
        'preresnet18_wd2',
        'preresnet18_w3d4',
        'preresnet18',
        'preresnet26',
        'preresnetbc26b',
        'preresnet34',
        'preresnetbc38b',
        'preresnet50',
        'preresnet50b',
        'preresnet101',
        'preresnet101b',
        'preresnet152',
        'preresnet152b',
        'preresnet200',
        'preresnet200b',
        'preresnet269b',
        'resnext14_16x4d',
        'resnext14_32x2d',
        'resnext14_32x4d',
        'resnext26_16x4d',
        'resnext26_32x2d',
        'resnext26_32x4d',
        'resnext38_32x4d',
        'resnext50_32x4d',
        'resnext101_32x4d',
        'resnext101_64x4d',
        'seresnet10',
        'seresnet12',
        'seresnet14',
        'seresnet16',
        'seresnet18',
        'seresnet26',
        'seresnetbc26b',
        'seresnet34',
        'seresnetbc38b',
        'seresnet50',
        'seresnet50b',
        'seresnet101',
        'seresnet101b',
        'seresnet152',
        'seresnet152b',
        'seresnet200',
        'seresnet200b',
        'sepreresnet10',
        'sepreresnet12',
        'sepreresnet14',
        'sepreresnet16',
        'sepreresnet18',
        'sepreresnet26',
        'sepreresnetbc26b',
        'sepreresnet34',
        'sepreresnetbc38b',
        'sepreresnet50',
        'sepreresnet50b',
        'sepreresnet101',
        'sepreresnet101b',
        'sepreresnet152',
        'sepreresnet152b',
        'sepreresnet200',
        'sepreresnet200b',
        'seresnext50_32x4d',
        'seresnext101_32x4d',
        'seresnext101_64x4d',
        'senet16',
        'senet28',
        'senet40',
        'senet52',
        'senet103',
        'senet154',
        'resnestabc14',
        'resnesta18',
        'resnestabc26',
        'resnesta50',
        'resnesta101',
        'resnesta152',
        'resnesta200',
        'resnesta269',
        'ibn_resnet50',
        'ibn_resnet101',
        'ibn_resnet152',
        'ibnb_resnet50',
        'ibnb_resnet101',
        'ibnb_resnet152',
        'ibn_resnext50_32x4d',
        'ibn_resnext101_32x4d',
        'ibn_resnext101_64x4d',
        'ibn_densenet121',
        'ibn_densenet161',
        'ibn_densenet169',
        'ibn_densenet201',
        'airnet50_1x64d_r2',
        'airnet50_1x64d_r16',
        'airnet101_1x64d_r2',
        'airnext50_32x4d_r2',
        'airnext101_32x4d_r2',
        'airnext101_32x4d_r16',
        'bam_resnet18',
        'bam_resnet34',
        'bam_resnet50',
        'bam_resnet101',
        'bam_resnet152',
        'cbam_resnet18',
        'cbam_resnet34',
        'cbam_resnet50',
        'cbam_resnet101',
        'cbam_resnet152',
        'resattnet56',
        'resattnet92',
        'resattnet128',
        'resattnet164',
        'resattnet200',
        'resattnet236',
        'resattnet452',
        'sknet50',
        'sknet101',
        'sknet152',
        'scnet50',
        'scnet101',
        'scneta50',
        'scneta101',
        'regnetx002',
        'regnetx004',
        'regnetx006',
        'regnetx008',
        'regnetx016',
        'regnetx032',
        'regnetx040',
        'regnetx064',
        'regnetx080',
        'regnetx120',
        'regnetx160',
        'regnetx320',
        'regnety002',
        'regnety004',
        'regnety006',
        'regnety008',
        'regnety016',
        'regnety032',
        'regnety040',
        'regnety064',
        'regnety080',
        'regnety120',
        'regnety160',
        'regnety320',
        'diaresnet10',
        'diaresnet12',
        'diaresnet14',
        'diaresnetbc14b',
        'diaresnet16',
        'diaresnet18',
        'diaresnet26',
        'diaresnetbc26b',
        'diaresnet34',
        'diaresnetbc38b',
        'diaresnet50',
        'diaresnet50b',
        'diaresnet101',
        'diaresnet101b',
        'diaresnet152',
        'diaresnet152b',
        'diaresnet200',
        'diaresnet200b',
        'diapreresnet10',
        'diapreresnet12',
        'diapreresnet14',
        'diapreresnetbc14b',
        'diapreresnet16',
        'diapreresnet18',
        'diapreresnet26',
        'diapreresnetbc26b',
        'diapreresnet34',
        'diapreresnetbc38b',
        'diapreresnet50',
        'diapreresnet50b',
        'diapreresnet101',
        'diapreresnet101b',
        'diapreresnet152',
        'diapreresnet152b',
        'diapreresnet200',
        'diapreresnet200b',
        'diapreresnet269b',
        'pyramidnet101_a360',
        'diracnet18v2',
        'diracnet34v2',
        'sharesnet18',
        'sharesnet34',
        'sharesnet50',
        'sharesnet50b',
        'sharesnet101',
        'sharesnet101b',
        'sharesnet152',
        'sharesnet152b',
        'densenet121',
        'densenet161',
        'densenet169',
        'densenet201',
        'condensenet74_c4_g4',
        'condensenet74_c8_g8',
        'sparsenet121',
        'sparsenet161',
        'sparsenet169',
        'sparsenet201',
        'sparsenet264',
        'peleenet',
        'wrn50_2',
        'drnc26',
        'drnc42',
        'drnc58',
        'drnd22',
        'drnd38',
        'drnd54',
        'drnd105',
        'dpn68',
        'dpn68b',
        'dpn98',
        'dpn107',
        'dpn131',
        'darknet_ref',
        'darknet_tiny',
        'darknet19',
        'darknet53',
        'channelnet',
        'revnet38',
        'revnet110',
        'revnet164',
        'irevnet301',
        'bagnet9',
        'bagnet17',
        'bagnet33',
        'dla34',
        'dla46c',
        'dla46xc',
        'dla60',
        'dla60x',
        'dla60xc',
        'dla102',
        'dla102x',
        'dla102x2',
        'dla169',
        'msdnet22',
        'fishnet99',
        'fishnet150',
        'espnetv2_wd2',
        'espnetv2_w1',
        'espnetv2_w5d4',
        'espnetv2_w3d2',
        'espnetv2_w2',
        'dicenet_wd5',
        'dicenet_wd2',
        'dicenet_w3d4',
        'dicenet_w1',
        'dicenet_w5d4',
        'dicenet_w3d2',
        'dicenet_w7d8',
        'dicenet_w2',
        'hrnet_w18_small_v1',
        'hrnet_w18_small_v2',
        'hrnetv2_w18',
        'hrnetv2_w30',
        'hrnetv2_w32',
        'hrnetv2_w40',
        'hrnetv2_w44',
        'hrnetv2_w48',
        'hrnetv2_w64',
        'vovnet27s',
        'vovnet39',
        'vovnet57',
        'selecsls42',
        'selecsls42b',
        'selecsls60',
        'selecsls60b',
        'selecsls84',
        'hardnet39ds',
        'hardnet68ds',
        'hardnet68',
        'hardnet85',
        'xdensenet121_2',
        'xdensenet161_2',
        'xdensenet169_2',
        'xdensenet201_2',
        'squeezenet_v1_0',
        'squeezenet_v1_1',
        'squeezeresnet_v1_0',
        'squeezeresnet_v1_1',
        'sqnxt23_w1',
        'sqnxt23_w3d2',
        'sqnxt23_w2',
        'sqnxt23v5_w1',
        'sqnxt23v5_w3d2',
        'sqnxt23v5_w2',
        'shufflenet_g1_w1',
        'shufflenet_g2_w1',
        'shufflenet_g3_w1',
        'shufflenet_g4_w1',
        'shufflenet_g8_w1',
        'shufflenet_g1_w3d4',
        'shufflenet_g3_w3d4',
        'shufflenet_g1_wd2',
        'shufflenet_g3_wd2',
        'shufflenet_g1_wd4',
        'shufflenet_g3_wd4',
        'shufflenetv2_wd2',
        'shufflenetv2_w1',
        'shufflenetv2_w3d2',
        'shufflenetv2_w2',
        'shufflenetv2b_wd2',
        'shufflenetv2b_w1',
        'shufflenetv2b_w3d2',
        'shufflenetv2b_w2',
        'menet108_8x1_g3',
        'menet128_8x1_g4',
        'menet160_8x1_g8',
        'menet228_12x1_g3',
        'menet256_12x1_g4',
        'menet348_12x1_g3',
        'menet352_12x1_g8',
        'menet456_24x1_g3',
        'mobilenet_w1',
        'mobilenet_w3d4',
        'mobilenet_wd2',
        'mobilenet_wd4',
        'mobilenetb_w1',
        'mobilenetb_w3d4',
        'mobilenetb_wd2',
        'mobilenetb_wd4',
        'fdmobilenet_w1',
        'fdmobilenet_w3d4',
        'fdmobilenet_wd2',
        'fdmobilenet_wd4',
        'mobilenetv2_w1',
        'mobilenetv2_w3d4',
        'mobilenetv2_wd2',
        'mobilenetv2_wd4',
        'mobilenetv2b_w1',
        'mobilenetv2b_w3d4',
        'mobilenetv2b_wd2',
        'mobilenetv2b_wd4',
        'mobilenetv3_small_w7d20',
        'mobilenetv3_small_wd2',
        'mobilenetv3_small_w3d4',
        'mobilenetv3_small_w1',
        'mobilenetv3_small_w5d4',
        'mobilenetv3_large_w7d20',
        'mobilenetv3_large_wd2',
        'mobilenetv3_large_w3d4',
        'mobilenetv3_large_w1',
        'mobilenetv3_large_w5d4',
        'igcv3_w1',
        'igcv3_w3d4',
        'igcv3_wd2',
        'igcv3_wd4',
        'ghostnet',
        'mnasnet_b1',
        'mnasnet_a1',
        'mnasnet_small',
        'darts',
        'proxylessnas_cpu',
        'proxylessnas_gpu',
        'proxylessnas_mobile',
        'proxylessnas_mobile14',
        'fbnet_cb',
        'nasnet_4a1056',
        'spnasnet',
        'efficientnet_b0',
        'efficientnet_b1',
        'efficientnet_b2',
        'efficientnet_b3',
        'efficientnet_b4',
        'efficientnet_b5',
        'efficientnet_b6',
        'efficientnet_b7',
        'efficientnet_b8',
        'efficientnet_b0b',
        'efficientnet_b1b',
        'efficientnet_b2b',
        'efficientnet_b3b',
        'efficientnet_b4b',
        'efficientnet_b5b',
        'efficientnet_b6b',
        'efficientnet_b7b',
        'efficientnet_b0c',
        'efficientnet_b1c',
        'efficientnet_b2c',
        'efficientnet_b3c',
        'efficientnet_b4c',
        'efficientnet_b5c',
        'efficientnet_b6c',
        'efficientnet_b7c',
        'efficientnet_b8c',
        'efficientnet_edge_small_b',
        'efficientnet_edge_medium_b',
        'efficientnet_edge_large_b',
        'mixnet_s',
        'mixnet_m',
        'mixnet_l',
        'isqrtcovresnet18',
        'isqrtcovresnet34',
        'isqrtcovresnet50',
        'isqrtcovresnet50b',
        'isqrtcovresnet101',
        'isqrtcovresnet101b',
        'resneta10',
        'resnetabc14b',
        'resneta18',
        'resneta50b',
        'resneta101b',
        'resneta152b',
        'resnetd50b',
        'resnetd101b',
        'resnetd152b',
        'fastseresnet101b',
        'octresnet10_ad2',
        'octresnet50b_ad2'
    ]
}


INPLACE_ABN_MODELS = [
    'densenet264d_iabn',
    'ese_vovnet99b_iabn',
    'tresnet_l_448',
    'tresnet_l',
    'tresnet_m_448',
    'tresnet_m',
    'tresnet_m_miil_in21k',
    'tresnet_v2_l',
    'tresnet_xl_448',
    'tresnet_xl'
]


# Models that can't handle 224x224 input size
FIXED_SIZE_INPUT_MODELS = [
    'bat_resnext26ts',
    'beit_base_patch16_384',
    'beit_large_patch16_384',
    'beit_large_patch16_512',
    'botnet26t_256',
    'botnet50ts_256',
    'cait_m36_384',
    'cait_m48_448',
    'cait_s24_384',
    'cait_s36_384',
    'cait_xs24_384',
    'cait_xxs24_384',
    'cait_xxs36_384',
    'deit3_base_patch16_384',
    'deit3_base_patch16_384_in21ft1k',
    'deit3_large_patch16_384',
    'deit3_large_patch16_384_in21ft1k',
    'deit3_small_patch16_384',
    'deit3_small_patch16_384_in21ft1k',
    'deit_base_distilled_patch16_384',
    'deit_base_patch16_384',
    'eca_botnext26ts_256',
    'eca_halonext26ts',
    'halo2botnet50ts_256',
    'halonet26t',
    'halonet50ts',
    'halonet_h1',
    'lambda_resnet26rpt_256',
    'lamhalobotnet50ts_256',
    'maxvit_nano_rw_256',
    'maxvit_pico_rw_256',
    'maxvit_rmlp_nano_rw_256',
    'maxvit_rmlp_pico_rw_256',
    'maxvit_rmlp_small_rw_256',
    'maxvit_rmlp_tiny_rw_256',
    'maxvit_tiny_pm_256',
    'maxvit_tiny_rw_256',
    'maxxvit_nano_rw_256',
    'maxxvit_small_rw_256',
    'maxxvit_tiny_rw_256',
    'sebotnet33ts_256',
    'sehalonet33ts',
    'swin_base_patch4_window12_384',
    'swin_base_patch4_window12_384_in22k',
    'swin_large_patch4_window12_384',
    'swin_large_patch4_window12_384_in22k',
    'swinv2_base_window12_192_22k',
    'swinv2_base_window12to16_192to256_22kft1k',
    'swinv2_base_window12to24_192to384_22kft1k',
    'swinv2_base_window16_256',
    'swinv2_base_window8_256',
    'swinv2_cr_base_384',
    'swinv2_cr_giant_384',
    'swinv2_cr_huge_384',
    'swinv2_cr_large_384',
    'swinv2_cr_small_384',
    'swinv2_cr_tiny_384',
    'swinv2_large_window12_192_22k',
    'swinv2_large_window12to16_192to256_22kft1k',
    'swinv2_large_window12to24_192to384_22kft1k',
    'swinv2_small_window16_256',
    'swinv2_small_window8_256',
    'swinv2_tiny_window16_256',
    'swinv2_tiny_window8_256',
    'vit_base_patch16_384',
    'vit_base_patch16_plus_240',
    'vit_base_patch32_384',
    'vit_base_patch32_plus_256',
    'vit_base_r50_s16_384',
    'vit_base_resnet50_384',
    'vit_large_patch16_384',
    'vit_large_patch32_384',
    'vit_large_r50_s32_384',
    'vit_relpos_base_patch16_plus_240',
    'vit_relpos_base_patch32_plus_rpn_256',
    'vit_small_patch16_384',
    'vit_small_patch32_384',
    'vit_small_r26_s32_384',
    'vit_tiny_patch16_384',
    'vit_tiny_r_s16_p8_384',
    'volo_d1_384',
    'volo_d2_384',
    'volo_d3_448',
    'volo_d4_448',
    'volo_d5_448',
    'volo_d5_512'
]
