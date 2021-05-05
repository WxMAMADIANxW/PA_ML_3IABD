#include "library.h"
#include <stdio.h>
#include <iostream>


#ifdef __WIN32__
#define DLLEXPORT _declspec(dllexport)
#else
#define DLLEXPORT
#endif

/*Présentation de la fonction
 * description
 * Prend en parametre ->
 * Retourne ->
 * */
DLLEXPORT <nom_fonction>(/*parametre*/) {
    /*Corp fonction*/
}
