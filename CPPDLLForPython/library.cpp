#include "library.h"
#include <stdio.h>
#include <iostream>
#include <Eigen>


#ifdef __WIN32__
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

/*Présentation de la fonction
 * description
 * Prend en parametre ->
 * Retourne ->
 * */
DLLEXPORT int toto() {
    return 1;
}
