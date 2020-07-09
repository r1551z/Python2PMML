```python
allButOne = ColumnTransformer([(str(cont_index), "passthrough", [cont_index]) for cont_index in range(46)]+
  [(str(cont_index), "passthrough", [cont_index]) for cont_index in range(47, 57)])

onlyOne = ColumnTransformer([(str(cont_index), "passthrough", [cont_index]) for cont_index in [46]])


estimator1=Pipeline(steps=[('Process', allButOne),
                          ('Estimator',
                           LGBMClassifier()
                          )
                          ]
                   )


estimator2=Pipeline(steps=[
    ('Process', onlyOne),
                          ('Estimator',LogisticRegression(multi_class='multinomial'))])


estimator = StackingClassifier([
  ("first", estimator1),
  ("second", estimator2),
  
], final_estimator = LogisticRegression(multi_class='multinomial'))


pipeline= PMMLPipeline([ ("domain", DataFrameMapper([
    (list(X.columns), ContinuousDomain(invalid_value_treatment ='as_is'))
  ])),

  ("ensemble", estimator_b)
                         ])
pipeline.fit(X_tv.iloc[:, :], y_tv.iloc[:])                         
pipeline.configure(compact = False, flat = False, winner_id = True)
sklearn2pmml(pipeline, "pipeline.pmml")
```
