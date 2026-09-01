(base) PS D:\Enterprise_AI_Knowledge_&_Operations_Copilot> my_env/Scripts/activate
(my_env) (base) PS D:\Enterprise_AI_Knowledge_&_Operations_Copilot> pip install python-dotenv
Collecting python-dotenv
  Downloading python_dotenv-1.2.3-py3-none-any.whl.metadata (29 kB)
Downloading python_dotenv-1.2.3-py3-none-any.whl (22 kB)
Installing collected packages: python-dotenv
Successfully installed python-dotenv-1.2.3

[notice] A new release of pip is available: 26.0.1 -> 26.2.1
[notice] To update, run: python.exe -m pip install --upgrade pip
(my_env) (base) PS D:\Enterprise_AI_Knowledge_&_Operations_Copilot> python generate_answer.py --index data\index --q "what is our vacation policy" --role hr
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|██████| 199/199 [00:00<00:00, 335.92it/s]
Direct use of automatic function calling (AFC) in Models.generate_content is not recommended. Instead, we recommend to use AFC in Chat.send_message. Similarly, direct use of AFC in Models.generate_content_stream is not recommended. Instead, we recommend to use AFC in Chat.send_message_stream.
Traceback (most recent call last):
  File "D:\Enterprise_AI_Knowledge_&_Operations_Copilot\generate_answer.py", line 202, in <module>
    main()
    ~~~~^^
  File "D:\Enterprise_AI_Knowledge_&_Operations_Copilot\generate_answer.py", line 190, in main
    answer, sources = generate_answer(args.index, args.q, args.role, args.provider, top_k=args.top_k)
                      ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Enterprise_AI_Knowledge_&_Operations_Copilot\generate_answer.py", line 174, in generate_answer
    answer = call_gemini(SYSTEM_INSTRUCTIONS, prompt)
  File "D:\Enterprise_AI_Knowledge_&_Operations_Copilot\generate_answer.py", line 126, in call_gemini
    response = client.models.generate_content(
        model="gemini-2.5-flash",
    ...<4 lines>...
        ),
    )
  File "D:\Enterprise_AI_Knowledge_&_Operations_Copilot\my_env\Lib\site-packages\google\genai\models.py", line 6270, in generate_content
    response = self._generate_content(
        model=model, contents=contents, config=parsed_config_to_call
    )
  File "D:\Enterprise_AI_Knowledge_&_Operations_Copilot\my_env\Lib\site-packages\google\genai\models.py", line 4707, in _generate_content
    response = self._api_client.request(
        'post', path, request_dict, http_options
    )
  File "D:\Enterprise_AI_Knowledge_&_Operations_Copilot\my_env\Lib\site-packages\google\genai\_api_client.py", line 1747, in request
    response = self._request(http_request, http_options, stream=False)
  File "D:\Enterprise_AI_Knowledge_&_Operations_Copilot\my_env\Lib\site-packages\google\genai\_api_client.py", line 1534, in _request
    return self._retry(self._request_once, http_request, stream)  # type: ignore[no-any-return]
           ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Enterprise_AI_Knowledge_&_Operations_Copilot\my_env\Lib\site-packages\tenacity\__init__.py", line 470, in __call__   
    do = self.iter(retry_state=retry_state)
  File "D:\Enterprise_AI_Knowledge_&_Operations_Copilot\my_env\Lib\site-packages\tenacity\__init__.py", line 371, in iter       
    result = action(retry_state)
  File "D:\Enterprise_AI_Knowledge_&_Operations_Copilot\my_env\Lib\site-packages\tenacity\__init__.py", line 413, in exc_check  
    raise retry_exc.reraise()
          ~~~~~~~~~~~~~~~~~^^
  File "D:\Enterprise_AI_Knowledge_&_Operations_Copilot\my_env\Lib\site-packages\tenacity\__init__.py", line 184, in reraise    
    raise self.last_attempt.result()
          ~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "E:\miniconda3\conda\Lib\concurrent\futures\_base.py", line 449, in result
    return self.__get_result()
           ~~~~~~~~~~~~~~~~~^^
  File "E:\miniconda3\conda\Lib\concurrent\futures\_base.py", line 401, in __get_result
    raise self._exception
  File "D:\Enterprise_AI_Knowledge_&_Operations_Copilot\my_env\Lib\site-packages\tenacity\__init__.py", line 473, in __call__   
    result = fn(*args, **kwargs)
  File "D:\Enterprise_AI_Knowledge_&_Operations_Copilot\my_env\Lib\site-packages\google\genai\_api_client.py", line 1511, in _request_once
    errors.APIError.raise_for_response(response)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^
  File "D:\Enterprise_AI_Knowledge_&_Operations_Copilot\my_env\Lib\site-packages\google\genai\errors.py", line 173, in raise_for_response
    cls.raise_error(response.status_code, response_json, response)
    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Enterprise_AI_Knowledge_&_Operations_Copilot\my_env\Lib\site-packages\google\genai\errors.py", line 202, in raise_error
    raise ClientError(status_code, response_json, response)     
google.genai.errors.ClientError: 400 INVALID_ARGUMENT. {'error': {'code': 400, 'message': 'API key not valid. Please pass a valid API key.', 'status': 'INVALID_ARGUMENT', 'details': [{'@type': 'type.googleapis.com/google.rpc.ErrorInfo', 'reason': 'API_KEY_INVALID', 'domain': 'googleapis.com', 'metadata': {'service': 'generativelanguage.googleapis.com'}}, {'@type': 'type.googleapis.com/google.rpc.LocalizedMessage', 'locale': 'en-US', 'message': 'API key not valid. Please pass a valid API key.'}]}}