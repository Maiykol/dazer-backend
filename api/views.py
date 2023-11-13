from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.views import APIView, View
from rest_framework.parsers import MultiPartParser, FileUploadParser, JSONParser
from rest_framework.decorators import parser_classes, api_view
from pathlib import Path
import os
import dazer
import pandas as pd
from django.http import HttpResponseBadRequest
from api import models, serializers, utils
import json
import numpy.random as npr
import shutil
from collections.abc import MutableMapping
from collections import defaultdict

from dazer_backend import tasks


@parser_classes([FileUploadParser])
class FileUpload(APIView):

    def put(self, request, session, filename):
        print(filename)
        content = request.body
        if content is None:
            return Response({})
        
        session_obj, _ = models.Session.objects.get_or_create(session_id=session)
        
        if len(models.File.objects.filter(session=session_obj, filename=filename)):
            # file already exists, increment filename
            for i in range(1, utils.ATTEMPTS+1):
                filename_incremented = f'{filename[:-4]}{i}.tsv'
                if not len(models.File.objects.filter(session=session_obj, filename=filename_incremented)):
                    break
            if i == utils.ATTEMPTS:
                return HttpResponseBadRequest('Filename already taken.')
            filename = filename_incremented
        
        Path(utils.get_session_files_folder(session)).mkdir(parents=True, exist_ok=True)
        file_path = os.path.join(utils.get_session_files_folder(session), filename)
        
        print(file_path)
        
        with open(file_path, "wb+") as destination:
            destination.write(content)
            
        try:
            df = utils.read_file(file_path)
            df, rows_removed = utils.clean_input_dataframe(df)
            columns, categorical_columns_values = utils.get_df_column_information(df)
        except:
            print('Could not read file')
            return HttpResponseBadRequest('Could not read file.')
        print(file_path)
        
        try:
            utils.write_file(df, file_path)
        except:
            print('Could not write file')
            return HttpResponseBadRequest('Could not write file.')
        
        print(file_path)
        
        # create session instance on file upload
        try:
            models.File.objects.create(session=session_obj, filename=filename, rows_removed=rows_removed, columns=json.dumps(columns), categorical_columns_values=json.dumps(categorical_columns_values))
        except Exception as e:
            print(e)
            print('Could not create file object')
            # file already exists for this session
            return HttpResponseBadRequest('File already exists.')
        
        print(file_path)
        
        
        return Response({})
    

class File(APIView):
    def get(self, request, subsample_id):
        subsample_obj = models.Subsampling.objects.get(subsample_id=subsample_id)
        file = serializers.FileSerializer().to_representation(subsample_obj.file)
        return Response({'file': file})
    
    
class FileColumns(APIView):
    def get(self, request, session, filename):
        file_obj = models.File.objects.get(session__session_id=session, filename=filename)
        return Response({'columns': json.loads(file_obj.columns), 'categorical_columns_values': json.loads(file_obj.categorical_columns_values)})
    

class FileColumnsSubsample(APIView):
    def get(self, request, subsample_id):
        subsample_obj = models.Subsampling.objects.get(subsample_id=subsample_id)
        file_obj = subsample_obj.file
        return Response({'columns': json.loads(file_obj.columns), 'categorical_columns_values': json.loads(file_obj.categorical_columns_values)})


class SessionId(APIView):
    # Session instance is created only on file upload
    
    def get(self, request):
        session = None
        while True:
            session = utils.generate_id(10)
            if not os.path.isdir(os.path.join("_sessions", session)):
                break

        if session is None:
            return Response({'session': ''})

        return Response({'session': session})
    
    
class SessionFiles(APIView):
    
    def get(self, request, session):
        session_files_folder = utils.get_session_files_folder(session)
        if not os.path.isdir(session_files_folder):
            return Response({'sessionFiles': []})
        file_obj_list = models.File.objects.filter(session__session_id=session).order_by('-created')
        session_files = serializers.FileSerializer(many=True).to_representation(file_obj_list)
        
        session_file_data = []
        for session_file in session_files:
            subsample_task_obj_list = models.Subsampling.objects.filter(session__session_id=session, file__filename=session_file['filename']).order_by('-created')
            
            subsample_task_information = []
            for subsample_task_obj in subsample_task_obj_list:
                classification_obj_list = models.ClassificationTask.objects.filter(subsample=subsample_task_obj).order_by('-created')
                classification_task_information = []
                for classification_task_obj in classification_obj_list:
                    classification_task_information.append({
                        'classificationId': classification_task_obj.classification_task_id, 'timestamp': classification_task_obj.created.timestamp(), 'progress': 1, 'error': False
                    })
                # add unfinished tasks
                classification_task_obj_list = models.Task.objects.filter(token__startswith=subsample_task_obj.subsample_id).filter(done=False).order_by('-created')
                for classification_task_obj in classification_task_obj_list:
                    classification_task_information.append({
                        'classificationId': classification_task_obj.token.split('_')[1], 'timestamp': classification_task_obj.created.timestamp(), 'progress': classification_task_obj.progress, 'error': classification_task_obj.failed
                    })
                 
                subsample_task_information.append({
                    'subsampleTaskId': subsample_task_obj.subsample_id,
                    'timestamp': subsample_task_obj.created.timestamp(),
                    'classificationTasks': classification_task_information
                    })

            session_file_data.append({
                **session_file, 
                'subsampleTasks': subsample_task_information
            })
                
        return Response({'sessionFiles': session_file_data})


def flatten_dict(dictionary, parent_key='', separator='_'):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + str(key) if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten_dict(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)
        
def merge_flat_dicts(array_of_dicts):
    keys = array_of_dicts[0].keys()
    return {key: sum([e[key] for e in array_of_dicts])/len(array_of_dicts) for key in keys}


@parser_classes([JSONParser])
class Subsample(APIView):
    
    def post(self, request, session, filename):
        random_states = [101, 102, 103, 104, 105]
        
        session_obj = models.Session.objects.get(session_id=session)
        file_obj = models.File.objects.get(session=session_obj, filename=filename)
        
        keep_ratio_columns = request.data.get('keepRatioColumns')
        test_ratio = request.data.get('testRatio')
        ratios = request.data.get('ratios', [.5, 1])
        n_random_states = int(request.data.get('nRandomStates', 1))
        allowed_deviation = float(request.data.get('allowedDeviation', 0.2))

        iteration_random_states = random_states[:n_random_states]
        
        #### move this to standalone task
        for _ in range(utils.ATTEMPTS):
            subsample_id = utils.generate_id(10)
            if not len(models.Subsampling.objects.filter(subsample_id=subsample_id)):
                break
            
        folder_test = utils.get_session_subsample_test_folder(session, filename, subsample_id)
        Path(folder_test).mkdir(parents=True, exist_ok=True)
        
        df = utils.read_file(os.path.join(utils.get_session_files_folder(session), filename))
        subsampler = dazer.Subsampler(df, keep_ratio_columns, allowed_deviation=allowed_deviation)
        for random_state in range(1, utils.ATTEMPTS+1):
            df_test = subsampler.extract_test(test_size=0.2, random_state=random_state)
            if df_test is not None:
                break
        df_test.to_csv(os.path.join(folder_test, f'type=test;ratio={str(test_ratio)};random_state={random_state}.tsv'), sep='\t')
        
        folder_train = utils.get_session_subsample_train_folder(session, filename, subsample_id)
        Path(folder_train).mkdir(parents=True, exist_ok=True)
        
        for seed in iteration_random_states:
            npr.seed(seed)
            for attempt in range(1, utils.ATTEMPTS+1):
                random_state = npr.randint(1, 999999999)
                for ratio in ratios:
                    df_train = subsampler.subsample(ratio, random_state)
                    if df_train is None:
                        # jump to next random state
                        break
                    df_train.to_csv(os.path.join(folder_train, f'type=train;ratio={str(ratio)};iteration_random_state={seed};random_state={random_state}.tsv'), sep='\t')
                if ratio == 1:
                    break
                
        ### SUBSAMPLING DONE ABOVE, FORMAT OUTPUT BELOW
        
        # train data
        folder_train = utils.get_session_subsample_train_folder(session, filename, subsample_id)
        data_all_random_iterations = []
        for iteration_state in iteration_random_states:
            ratios_data = {}
            for file in os.listdir(folder_train):
                if f'iteration_random_state={iteration_state}' not in file:
                    continue
                for param in file.split(';'):
                    key, value = param.split('=')
                    if key == 'ratio':
                        ratios_data[value] = file
            data = {}
            for ratio, path in ratios_data.items():
                df = utils.read_file(os.path.join(folder_train, path))
                data[f'train:{ratio}'] = {}
                for col in keep_ratio_columns:
                    data[f'train:{ratio}'][col] = df[col].value_counts().to_dict()
                        
            # test data
            folder_test = utils.get_session_subsample_test_folder(session, filename, subsample_id)
            file = os.listdir(folder_test)[0]
            test_ratio = ''
            for param in file.split(';'):
                key, value = param.split('=')
                if key == 'ratio':
                    test_ratio = value
            df = utils.read_file(os.path.join(folder_test, file))
            data[f'test:{test_ratio}'] = {}
            for col in keep_ratio_columns:
                data[f'test:{test_ratio}'][col] = df[col].value_counts().to_dict()
            
            # reformat data for plotting
            data_reformat = {col: {} for col in keep_ratio_columns}
            for ratio_category, ratio_data in data.items():
                for keep_ratio_col, value_counts in ratio_data.items():
                    if 'test' in ratio_category:
                        data_reformat[keep_ratio_col][ratio_category] = value_counts
                    else:
                        ratio = ratio_category.split(':')[1]
                        data_reformat[keep_ratio_col][ratio] = value_counts
            data_all_random_iterations.append(data_reformat)
            
        mean_dict = defaultdict(lambda: defaultdict(dict))
        for key, mean in merge_flat_dicts([flatten_dict(d) for d in data_all_random_iterations]).items():
            feature, ratio, value = key.split('_')
            mean_dict[feature][ratio][value] = mean
        
        # get categorical columns
        categorical_columns = df.columns[~df.columns.isin(df._get_numeric_data().columns)]
        unique_values = {}
        for col in categorical_columns:
            unique_values[col] = df[col].dropna().unique().tolist()
            
        ### RESULT FORMATTING DONE
        
        ### END move this to standalone task
        
        # test_file = os.listdir(get_session_subsample_test_folder(session, filename, subsample_id))
        # train_files = os.listdir(get_session_subsample_train_folder(session, filename, subsample_id))
            
        models.Subsampling.objects.create(
            session=session_obj,
            file=file_obj,
            subsample_id=subsample_id,
            keep_ratio_columns=json.dumps(keep_ratio_columns),
            ratios=json.dumps(ratios_data),
            iteration_random_states=json.dumps(iteration_random_states),
            test_ratio=test_ratio,
            allowed_deviation=allowed_deviation,
            result_formatted =json.dumps({'data': dict(mean_dict), 'filename': filename, 'keepRatioColumns': keep_ratio_columns, 'ratios': ratios, 'testLabel': f'test:{test_ratio}'})
            )
            
        return Response()
        

class SubsampleResult(APIView):
    
    def get(self, request, subsample_id):
        subsample_obj = models.Subsampling.objects.get(subsample_id=subsample_id)
        return Response({**json.loads(subsample_obj.result_formatted), 'allowedDeviation': subsample_obj.allowed_deviation, 'iterationRandomStates': json.loads(subsample_obj.iteration_random_states), 'testRatio': subsample_obj.test_ratio})
    
        
@parser_classes([JSONParser])
class Classification(APIView):
    
    def post(self, request, subsample_id):
        random_states = [101, 102, 103, 104, 105]
        
        classification_task_id = ''
        for _ in range(utils.ATTEMPTS):
            classification_task_id = utils.generate_id(10)
            if not len(models.ClassificationTask.objects.filter(classification_task_id=classification_task_id)):
                break
        assert len(classification_task_id)
        
        subsample_obj = models.Subsampling.objects.get(subsample_id=subsample_id)
        target_column = request.data.get('targetColumn')
        target_value = request.data.get('targetValue')
        cv = int(request.data.get('crossValidationK', 2))
        n_random_states = int(request.data.get('nRandomStates', 1))
        
        if not (1 <= n_random_states <= 5):
            return HttpResponseBadRequest('nRandomStates must be a value between 1 and 5')
        random_states = random_states[:n_random_states]

        if not (1 <= cv <= 10):
            return HttpResponseBadRequest('crossValidationK must be a value between 1 and 10')
        
        task_token = f'{subsample_id}_{classification_task_id}'
        task = models.Task.objects.create(
            token=f'{subsample_id}_{classification_task_id}',
            objective='subsample',
            parameters=json.dumps({
                'target_column': target_column,
                'target_value': target_value,
                'cv': cv,
                'random_states': random_states
                }),
        )
        
        # add task to asynchronous queue
        tasks.classification_task.delay(task_token)
        return Response({'classification_task_id': classification_task_id})
    
    
def _merge_classification_results(json_string):
    # merge results of different random to calculate average in backend since it is more convenient here.
    # Is there a nice way to visualize the single points of the random states in the frontend?
    array = json.loads(json_string)
    df = pd.DataFrame(array)
    return df.groupby('ratio').mean().reset_index().to_dict(orient='records')
    
class ClassificationResult(APIView):    
    
    def get(self, request, classification_task_id):
        
        task_obj = models.Task.objects.get(token__endswith=classification_task_id)
        print('task_obj', task_obj)
        print(task_obj.done)
        print(task_obj.failed)
        if not (task_obj.done or task_obj.failed):
            return Response({
                'status': json.loads(task_obj.status) if task_obj.status is not None else '',
                'progress': task_obj.progress
            })
        
        print('here')
        classification_task = models.ClassificationTask.objects.get(classification_task_id=classification_task_id)
        print('classification_task', classification_task)
        evaluation_list = _merge_classification_results(classification_task.evaluation)
        feature_importances = json.loads(classification_task.feature_importances)
        feature_importances_features = json.loads(classification_task.feature_columns)
        
        if len(feature_importances_features) > 50:
            # get most important features
            df_feature_importances = pd.DataFrame(feature_importances, columns=feature_importances_features)
            feature_importances_features = df_feature_importances.mean().sort_values(ascending=False).index[:50]
            feature_importances = [list(x) for x in df_feature_importances[feature_importances_features].values]
            
        data = json.loads(classification_task.evaluation)
        data = sorted(data,  key=lambda d: d['ratio'])
        return Response({
            'data_merged': evaluation_list,
            'data': data,
            'feature_importances': feature_importances,
            'filename': classification_task.subsample.file.filename,
            'target_column': classification_task.target_column,
            'target_value': classification_task.target_value,
            'feature_importances_features': feature_importances_features,
            'crossValidationK': classification_task.cv,
            'randomStates': json.loads(classification_task.random_states),
            })


def delete_classification_task(classification_task_id):
    classification_task = models.ClassificationTask.objects.get(classification_task_id=classification_task_id)
    model_folder = utils.get_model_folder(classification_task.subsample.session.session_id, classification_task.subsample.file.filename, classification_task_id)
    # delete folder with models
    shutil.rmtree(model_folder)
    # delete db entry
    classification_task.delete()
    
    
class ClassificationDelete(APIView):    
    
    def get(self, request, classification_task_id):
        delete_classification_task(classification_task_id)
        return Response()
        

def delete_subsample_task(subsample_id):
    subsample_obj = models.Subsampling.objects.get(subsample_id=subsample_id)
    
    for classifcation_task in subsample_obj.classification:
        delete_classification_task(classifcation_task.classification_task_id)
    
    subsample_folder = utils.get_session_subsample_task_folder(subsample_obj.session.session_id, subsample_obj.file.filename, subsample_id)
    # delete folder with subsample results
    shutil.rmtree(subsample_folder)
    # delete db entry
    subsample_obj.delete()


class SubsampleDelete(APIView):    
    
   def get(self, request, subsample_id):
        delete_subsample_task(subsample_id)
        return Response()
    
    
def delete_file(session, filename):
    file_obj = models.File.objects.get(session__session_id=session, filename=filename)
    
    for subsample in file_obj.subsamples:
        delete_subsample_task(subsample.subsample_id)
    
    session_files_folder = utils.get_session_files_folder(session)
    # delete delete uploaded file
    os.remove(os.path.join(session_files_folder, filename))
    # delete db entry
    file_obj.delete()
    
    
class FileDelete(APIView):    
    
    def get(self, request, session, filename):
        delete_file(session, filename)
        return Response()
    

def delete_session(session):
    session_obj = models.Session.objects.get(session_id=session)
    
    for file_obj in session_obj.files:
        delete_file(file_obj.filename)
    
    session_files_folder = utils.get_session_files_folder(session)
    # delete delete all uploaded files
    shutil.rmtree(session_files_folder)
    # delete db entry
    session_obj.delete()


class SessionDelete(APIView):    
    
    def get(self, request, session):
        delete_session(session)
        return Response()