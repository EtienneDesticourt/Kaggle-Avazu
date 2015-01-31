/*
 * generatorFunctions.c
 *
 *  Created on: 31 Jan 2015
 *      Author: Etienne
 */

#include <Python.h>



static void genPolynomialFeatures(PyObject *self, PyObject *args){

	char * tok;         /* delimiter tokens for strtok */
	int cols;           /* number of cols to parse, from the left */



	PyObject * features; /* the list of strings */
	char * feature0;  /* one string in the list */
	char * feature1;  /* one string in the list */
	PyObject * newFeature; /* string to append */
	//Parse list
	PyArg_ParseTuple( args, "O!is", &PyList_Type, &features,&cols, &tok ) ;

	int originalLength = PyList_Size(features);

	/* should raise an error here. */
	if (originalLength < 0)   return; /* Not a list */

	int i;
	int j;
	for (i=0; i<originalLength; i++){
		for (j=i+1; j<originalLength; j++){
			//Get the two features
			feature0 = PyString_AsString( PyList_GetItem(features, i) );
			feature1 = PyString_AsString( PyList_GetItem(features, j) );

			//Concatenate them
			char *result = malloc(strlen(feature0)+strlen("_")+strlen(feature1)+1);
			strcpy(result, feature0);
			strcat(result, "_");
			strcat(result, feature1);

			//Add them to the list
			newFeature = PyString_FromString(result);
			PyList_Append(features, newFeature);


		}
	}
}

static PyMethodDef GeneratorFunctionsMethods[] = {
    {"genPolyFeatures",  genPolynomialFeatures, METH_VARARGS,
     "Generates polynomial features."},
};

PyMODINIT_FUNC initgeneratorFunctions(void){
    (void) Py_InitModule("generatorFunctions", GeneratorFunctionsMethods);
}
