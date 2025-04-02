"""
This module contains various functions for symbolic regression,
which is accelerated through the utilization of parallel computing techniques.

The capabilities of the symbolic regression methods
— search_trivial_relations
- search_relations_ver1
- search_relations_ver2
- search_relations_ver3
are ranked in ascending order. The search_relations_ver3 method is the most powerful one,
while it takes a long time to run.
"""

from typing import List, Tuple, Dict, Callable
from aiphy.core import (
    search_trivial_relations,
    search_binary_relations,
    search_relations_ver1,
    search_relations_ver2,
    search_relations_ver3,
    DataStruct,
    Exp,
    ExpData
)

from .pca import search_relations_by_pca

def search_relations_ver3_and_pca(ds: DataStruct, debug: bool = False, cpu_num: int = 50) -> List[Tuple[Exp, ExpData]]:
    res = search_relations_ver3(ds, debug, cpu_num)
    res_pca = search_relations_by_pca(ds)
    res_dict = {i[0]: i[1] for i in res}
    res_pca_dict = {i[0]: i[1] for i in res_pca}
    res_dict.update(res_pca_dict)
    return [(i, res_dict[i]) for i in res_dict]

search_type: Dict[str, Callable[[DataStruct, bool, int], List[Tuple[Exp, ExpData]]]] = {
    None: search_relations_ver1,
    'trivial': search_trivial_relations,
    'ver1': search_relations_ver1,
    'ver2': search_relations_ver2,
    'ver3': search_relations_ver3_and_pca,
}
