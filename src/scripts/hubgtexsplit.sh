#!/usr/bin/bash
set -e
CUT_DIR="CUTS_1.0"
F_DIR="2.000_1024"
P=$PWD



OUT=$P/input/$CUT_DIR/gtex_test/kidney/$F_DIR/images
mkdir $OUT -p
cd $P/input/$CUT_DIR/gtex/kidney/$F_DIR/images/
mv 0_2262_4788_10748_11581_0000* 2_32749_2709_11388_12349_0000* 4_22032_22257_9085_11005_0000* 1_17489_3445_11420_11613_0000* 3_35980_19409_11068_11581_0000* 5_4661_25136_10012_10781_0000* $OUT

OUT=$P/input/$CUT_DIR/gtex_test/kidney/$F_DIR/masks
mkdir $OUT -p
cd $P/input/$CUT_DIR/gtex/kidney/$F_DIR/masks/
mv 0_2262_4788_10748_11581_0000* 2_32749_2709_11388_12349_0000* 4_22032_22257_9085_11005_0000* 1_17489_3445_11420_11613_0000* 3_35980_19409_11068_11581_0000* 5_4661_25136_10012_10781_0000* $OUT


OUT=$P/input/$CUT_DIR/gtex_test/spleen/$F_DIR/images
mkdir $OUT -p
cd $P/input/$CUT_DIR/gtex/spleen/$F_DIR/images/
mv 0_5271_8052_20187_21947_0000* 1_35184_6324_21467_19515_0000* 0_3574_4787_19611_21627_0000* 1_25137_6546_24154_22363_0000* $OUT

OUT=$P/input/$CUT_DIR/gtex_test/spleen/$F_DIR/masks
mkdir $OUT -p
cd $P/input/$CUT_DIR/gtex/spleen/$F_DIR/masks/
mv 0_5271_8052_20187_21947_0000* 1_35184_6324_21467_19515_0000* 0_3574_4787_19611_21627_0000* 1_25137_6546_24154_22363_0000* $OUT


OUT=$P/input/$CUT_DIR/gtex_test/colon/$F_DIR/images
mkdir $OUT -p
cd $P/input/$CUT_DIR/gtex/colon/$F_DIR/images/
mv 1_15028_25841_18556_12381_0000* 2_31025_6677_8158_7198_0000* 0_10645_8853_14908_16156_0000* $OUT

OUT=$P/input/$CUT_DIR/gtex_test/colon/$F_DIR/masks
mkdir $OUT -p
cd $P/input/$CUT_DIR/gtex/colon/$F_DIR/masks/
mv 1_15028_25841_18556_12381_0000* 2_31025_6677_8158_7198_0000* 0_10645_8853_14908_16156_0000* $OUT



OUT=$P/input/$CUT_DIR/hubmap_test/kidney/$F_DIR/images
mkdir $OUT -p
cd $P/input/$CUT_DIR/hubmap/kidney/$F_DIR/images/
mv 1e2425f28_0000* 2f6ecfcdf_0000* 4ef6695ce_0000* $OUT

OUT=$P/input/$CUT_DIR/hubmap_test/kidney/$F_DIR/masks
mkdir $OUT -p
cd $P/input/$CUT_DIR/hubmap/kidney/$F_DIR/masks/
mv 1e2425f28_0000* 2f6ecfcdf_0000* 4ef6695ce_0000* $OUT

