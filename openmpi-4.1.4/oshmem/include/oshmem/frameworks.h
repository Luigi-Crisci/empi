/*
 * This file is autogenerated by autogen.pl. Do not edit this file by hand.
 */
#ifndef OSHMEM_FRAMEWORKS_H
#define OSHMEM_FRAMEWORKS_H

#include <opal/mca/base/mca_base_framework.h>

extern mca_base_framework_t oshmem_atomic_base_framework;
extern mca_base_framework_t oshmem_memheap_base_framework;
extern mca_base_framework_t oshmem_scoll_base_framework;
extern mca_base_framework_t oshmem_spml_base_framework;
extern mca_base_framework_t oshmem_sshmem_base_framework;

static mca_base_framework_t *oshmem_frameworks[] = {
    &oshmem_atomic_base_framework,
    &oshmem_memheap_base_framework,
    &oshmem_scoll_base_framework,
    &oshmem_spml_base_framework,
    &oshmem_sshmem_base_framework,
    NULL
};

#endif /* OSHMEM_FRAMEWORKS_H */

