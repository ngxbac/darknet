UFPR_DATA=data/ufpr_alpr.data
UFPR_CFG=data/yolov4-lp-ufpr.cfg
CCPD_DATA=data/ccpd.data
CCPD_CFG=data/yolov4-lp-ccpd.cfg
MOT16_DATA=data/mot16.data
MOT16_CFG=data/yolov4-mot16.cfg
UFPR_CHAR_DATA=data/ufpr_char.data
UFPR_CHAR_CFG=data/yolov4-ufpr-char.cfg
VNSYN_DATA=data/vn_syn_plate.data
VNSYN_CFG=data/yolov4-vnsyn.cfg

# darknet detector test data/ccpd.data data/yolov4-lp-ccpd.cfg ./ccpd_sgdr_mixup_mosac/yolov4-lp_2000.weights -dont_show -ext_output < data/ufpr_alpr_training.txt > result.txt

train_ufpr:
	./darknet detector train ${UFPR_DATA} ${UFPR_CFG} -dont_show

anchors_ufpr:
	./darknet detector calc_anchors ${UFPR_DATA} -num_of_clusters 9 -width 416 -height 416

valid_ufpr:
	./darknet detector 	map \
						${UFPR_DATA} ${UFPR_CFG} \
						./ufpr_alpr_from_car_sgdr_mixup_mosac/yolov4-lp-ufpr_last.weights \
						-iou_thresh 0.5 ; \
	./darknet detector 	map \
						${UFPR_DATA} ${UFPR_CFG} \
						./ufpr_alpr_from_car_sgdr_mixup_mosac/yolov4-lp-ufpr_last.weights \
						-iou_thresh 0.75 ; \
	./darknet detector 	map \
						${UFPR_DATA} ${UFPR_CFG} \
						./ufpr_alpr_from_car_sgdr_mixup_mosac/yolov4-lp-ufpr_last.weights \
						-iou_thresh 0.95

train_cppd:
	./darknet detector train ${CCPD_DATA} ${CCPD_CFG} -dont_show

valid_ccpd:
	./darknet detector 	map \
						${CCPD_DATA} ${CCPD_CFG} \
						./ccpd_sgdr_mixup_mosac/yolov4-lp_last.weights \
						-iou_thresh 0.5 ; \
	./darknet detector 	map \
						${CCPD_DATA} ${CCPD_CFG} \
						./ccpd_sgdr_mixup_mosac/yolov4-lp_last.weights \
						-iou_thresh 0.75 ; \
	./darknet detector 	map \
						${CCPD_DATA} ${CCPD_CFG} \
						./ccpd_sgdr_mixup_mosac/yolov4-lp_last.weights \
						-iou_thresh 0.95

train_mot16:
	./darknet detector train ${MOT16_DATA} ${MOT16_CFG} -dont_show

anchors_mot16:
	./darknet detector calc_anchors ${MOT16_DATA} -num_of_clusters 9 -width 416 -height 416


train_ufpr_char:
	./darknet detector train ${UFPR_CHAR_DATA} ${UFPR_CHAR_CFG} -dont_show

valid_ufpr_char:
	./darknet detector map ${UFPR_CHAR_DATA} ${UFPR_CHAR_CFG} ufpr_char_128_sgdr_mixup_mosac/yolov4-ufpr-char_60000.weights -dont_show

anchors_ufpr_char:
	./darknet detector calc_anchors ${UFPR_CHAR_DATA} -num_of_clusters 9 -width 128 -height 128


train_vnsyn:
	./darknet detector train ${VNSYN_DATA} ${VNSYN_CFG} -dont_show

valid_vnsyn:
	./darknet detector 	map \
						${VNSYN_DATA} ${VNSYN_CFG} \
						./vn_syn_hard3/yolov4-vnsyn_last.weights \
						-iou_thresh 0.5 ; \
	./darknet detector 	map \
						${VNSYN_DATA} ${VNSYN_CFG} \
						./vn_syn_hard3/yolov4-vnsyn_last.weights \
						-iou_thresh 0.75 ; \
	./darknet detector 	map \
						${VNSYN_DATA} ${VNSYN_CFG} \
						./vn_syn_hard3/yolov4-vnsyn_last.weights \
						-iou_thresh 0.95

anchors_vnsyn:
	./darknet detector calc_anchors ${VNSYN_DATA} -num_of_clusters 9 -width 128 -height 128

#cross_valid_ufpr:
#	./darknet detector test ${CCPD_DATA} ${CCPD_CFG} \
#	 						./ccpd_sgdr_mixup_mosac/yolov4-lp_2000.weights \
#	 						-dont_show -ext_output < data/ufpr_alpr_training.txt > cv_ufpr_training_result.txt ; \
#	./darknet detector test ${CCPD_DATA} ${CCPD_CFG} \
#						./ccpd_sgdr_mixup_mosac/yolov4-lp_2000.weights \
#						-dont_show -ext_output < data/ufpr_alpr_validation.txt > cv_ufpr_validation_result.txt ; \
#	./darknet detector test ${CCPD_DATA} ${CCPD_CFG} \
#					./ccpd_sgdr_mixup_mosac/yolov4-lp_2000.weights \
#					-dont_show -ext_output < data/ufpr_alpr_testing.txt > cv_ufpr_testing_result.txt