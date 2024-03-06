default: pytest

# default: pylint pytest

# pylint:
# 	find . -iname "*.py" -not -path "./tests/test_*" | xargs -n1 -I {}  pylint --output-format=colorized {}; true

pytest:
	echo "no tests"

# ----------------------------------
#         LOCAL SET UP
# ----------------------------------

install_requirements:
	@pip install -r requirements.txt

# ----------------------------------
#         HEROKU COMMANDS
# ----------------------------------

streamlit:
	@streamlit run app.py


# ----------------------------------
#    LOCAL INSTALL COMMANDS
# ----------------------------------
install:
	@pip install . -U

clean:
	@rm -fr */__pycache__
	@rm -fr __init__.py
	@rm -fr build
	@rm -fr dist
	@rm -fr *.dist-info
	@rm -fr *.egg-info
	-@rm model.joblib



train_save_basic_model:
	python -c 'from base_fruit_classifier.main import train_save_basic_model; train_save_basic_model()'



####
.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y taxifare || :
	@pip install -e .

run_preprocess:
	python -c 'from taxifare.interface.main import preprocess; preprocess()'

run_train:
	python -c 'from base_fruit_classifier.main import train_save_basic_model; train_save_basic_model()'

run_train_save_basic_model: # already working
	@python3 -c 'from base_fruit_classifier.main import train_save_basic_model; train_save_basic_model()'

run_train_save_resnet50: # already working
	@python3 -c 'from base_fruit_classifier.main import train_save_resnet50; train_save_resnet50()'

run_train_save_basic_model_vm: # already working
	@python3 -c 'from base_fruit_classifier.main import train_save_basic_model; train_save_basic_model()'

run_load_model: # already working. Returns latest model trained either from local or GCP
	@python3 -c 'from base_fruit_classifier.registry import load_model; load_model()'

run_get_dataset_class_names: # already working
	@python3 -c 'from base_fruit_classifier.main import get_dataset_classes; get_dataset_classes()'

run_download_images_to_predict: # already working. Download images to predict from bucket GCP into local
	@python3 -c 'from base_fruit_classifier.registry import download_images_to_predict; download_images_to_predict()'

run_predict_vgg16: # already working. Predict images downloaded from GCP.
	@python3 -c 'from base_fruit_classifier.main import predict; predict(model_type="vgg16", img_height=348, img_width=348)'


run_predict_resnet50: # already working. Predict images downloaded from GCP.
	@python3 -c 'from base_fruit_classifier.main import predict; predict(model_type="resnet50", img_height=100, img_width=100)'


run_count_items_in_bucket_dataset: # already working.Return number of items in bucket dataset
	@python3 -c 'from base_fruit_classifier.registry import count_items_in_bucket_dataset; count_items_in_bucket_dataset()'


run_print_items_bucket_dataset: # already working. Prints names of items in bucket dataset
	@python3 -c 'from base_fruit_classifier.registry import print_items_in_bucket_dataset; print_items_in_bucket_dataset()'

run_download_training_dataset: # already working. Prints names of items in bucket dataset
	@python3 -c 'from base_fruit_classifier.registry import download_training_dataset; download_training_dataset()'

run_train_save_vgg16: # already working.
	@python3 -c 'from base_fruit_classifier.main import train_save_vgg16; train_save_vgg16()'



run_pred:
	python -c 'from taxifare.interface.main import pred; pred()'

run_evaluate:
	python -c 'from taxifare.interface.main import evaluate; evaluate()'

run_all: run_preprocess run_train run_pred run_evaluate

run_workflow:
	PREFECT__LOGGING__LEVEL=${PREFECT_LOG_LEVEL} python -m taxifare.interface.workflow

run_api:
	uvicorn taxifare.api.fast:app --reload

##################### TESTS #####################
test_gcp_setup:
	@pytest \
	tests/all/test_gcp_setup.py::TestGcpSetup::test_setup_key_env \
	tests/all/test_gcp_setup.py::TestGcpSetup::test_setup_key_path \
	tests/all/test_gcp_setup.py::TestGcpSetup::test_code_get_project \
	tests/all/test_gcp_setup.py::TestGcpSetup::test_code_get_wagon_project

default:
	cat tests/api/test_output.txt

test_kitt:
	@echo "\n 🧪 computing and saving your progress at 'tests/api/test_output.txt'..."
	@pytest tests/api -c "./tests/pytest_kitt.ini" 2>&1 > tests/api/test_output.txt || true
	@echo "\n 🙏 Please: \n git add tests \n git commit -m 'checkpoint' \n ggpush"

test_api_root:
	pytest \
	tests/api/test_endpoints.py::test_root_is_up --asyncio-mode=strict -W "ignore" \
	tests/api/test_endpoints.py::test_root_returns_greeting --asyncio-mode=strict -W "ignore"

test_api_predict:
	pytest \
	tests/api/test_endpoints.py::test_predict_is_up --asyncio-mode=strict -W "ignore" \
	tests/api/test_endpoints.py::test_predict_is_dict --asyncio-mode=strict -W "ignore" \
	tests/api/test_endpoints.py::test_predict_has_key --asyncio-mode=strict -W "ignore" \
	tests/api/test_endpoints.py::test_predict_val_is_float --asyncio-mode=strict -W "ignore"

test_api_on_prod:
	pytest \
	tests/api/test_cloud_endpoints.py --asyncio-mode=strict -W "ignore"


################### DATA SOURCES ACTIONS ################

# Data sources: targets for monthly data imports
#ML_DIR=~/.lewagon/mlops
#HTTPS_DIR=https://storage.googleapis.com/datascience-mlops/taxi-fare-ny/
#GS_DIR=gs://datascience-mlops/taxi-fare-ny
CM_DIR = ~/chillmate/

show_sources_all:
	-ls -laR ~/.lewagon/mlops/data
	-bq ls ${BQ_DATASET}
	-bq show ${BQ_DATASET}.processed_1k
	-bq show ${BQ_DATASET}.processed_200k
	-bq show ${BQ_DATASET}.processed_all
	-gsutil ls gs://${BUCKET_NAME}

reset_local_files:
	@rm -rf ${ML_DIR}
	@mkdir -p ~/chillmate/chillmate-models/
	@mkdir -p ~/chillmate/chillmate-models/vgg16
	@mkdir -p ~/chillmate/chillmate-models/resnet50
	@mkdir -p ~/chillmate/chillmate-models/basic
	@mkdir -p ~/chillmate/images-to-predict
	@mkdir -p ~/chillmate/dataset



reset_local_files_with_csv_solutions: reset_local_files
	-curl ${HTTPS_DIR}solutions/data_query_fixture_2009-01-01_2015-01-01_1k.csv > ${ML_DIR}/data/raw/query_2009-01-01_2015-01-01_1k.csv
	-curl ${HTTPS_DIR}solutions/data_query_fixture_2009-01-01_2015-01-01_200k.csv > ${ML_DIR}/data/raw/query_2009-01-01_2015-01-01_200k.csv
	-curl ${HTTPS_DIR}solutions/data_query_fixture_2009-01-01_2015-01-01_all.csv > ${ML_DIR}/data/raw/query_2009-01-01_2015-01-01_all.csv
	-curl ${HTTPS_DIR}solutions/data_processed_fixture_2009-01-01_2015-01-01_1k.csv > ${ML_DIR}/data/processed/processed_2009-01-01_2015-01-01_1k.csv
	-curl ${HTTPS_DIR}solutions/data_processed_fixture_2009-01-01_2015-01-01_200k.csv > ${ML_DIR}/data/processed/processed_2009-01-01_2015-01-01_200k.csv
	-curl ${HTTPS_DIR}solutions/data_processed_fixture_2009-01-01_2015-01-01_all.csv > ${ML_DIR}/data/processed/processed_2009-01-01_2015-01-01_all.csv

reset_bq_files:
	-bq rm --project_id ${GCP_PROJECT} ${BQ_DATASET}.processed_1k
	-bq rm --project_id ${GCP_PROJECT} ${BQ_DATASET}.processed_200k
	-bq rm --project_id ${GCP_PROJECT} ${BQ_DATASET}.processed_all
	-bq mk --sync --project_id ${GCP_PROJECT} --location=${BQ_REGION} ${BQ_DATASET}.processed_1k
	-bq mk --sync --project_id ${GCP_PROJECT} --location=${BQ_REGION} ${BQ_DATASET}.processed_200k
	-bq mk --sync --project_id ${GCP_PROJECT} --location=${BQ_REGION} ${BQ_DATASET}.processed_all

reset_gcs_files:
	-gsutil rm -r gs://${BUCKET_NAME}
	-gsutil mb -p ${GCP_PROJECT} -l ${GCP_REGION} gs://${BUCKET_NAME}

reset_all_files: reset_local_files reset_bq_files reset_gcs_files
