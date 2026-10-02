/**
 * @file _ctrie.c
 * @brief Fast Trie data structure implementation for the uztoken library.
 *
 * This C extension provides a high-performance Trie for Uzbek language processing,
 * supporting UTF-8 strings, Uzbek Latin characters with standard ASCII apostrophe ('),
 * Uzbek Cyrillic characters, and morphological suffix lookup.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define TRIE_CAPSULE_NAME "uztoken._ctrie.Trie"
#define MAX_COMPLETION_DEPTH 1024

/* Forward declarations */
typedef struct TrieNode TrieNode;
typedef struct UnicodeNode UnicodeNode;

/**
 * @brief Node for non-ASCII (Unicode codepoints >= 256) in a linked list.
 */
struct UnicodeNode {
    uint32_t codepoint;       /* Unicode codepoint (>= 256) */
    TrieNode *node;           /* Pointer to child TrieNode */
    UnicodeNode *next;        /* Next sibling in linked list */
};

/**
 * @brief Core Trie node structure.
 *
 * Memory optimization:
 * - ascii_children: dynamically allocated array of 256 TrieNode pointers.
 *   Allocated only when the node has at least one ASCII child (0-255).
 *   Leaf nodes do not allocate this array, saving substantial memory.
 * - unicode_children: singly-linked list for codepoints >= 256 (Cyrillic, etc.).
 */
struct TrieNode {
    int is_end;                       /* 1 if word terminates at this node, 0 otherwise */
    TrieNode **ascii_children;        /* Array of 256 pointers for ASCII/Latin-1 (0-255) */
    UnicodeNode *unicode_children;    /* Linked list for Unicode codepoints >= 256 */
};

/**
 * @brief Top-level Trie container.
 */
typedef struct {
    TrieNode *root;
    size_t word_count;
} Trie;

/* ============================================================================
 * UTF-8 Helper Functions
 * ============================================================================ */

/**
 * @brief Decodes the next Unicode codepoint from a UTF-8 string.
 *
 * @param str Pointer to current position in UTF-8 string.
 * @param codepoint Output pointer to receive decoded 32-bit codepoint.
 * @return Pointer to the beginning of the next UTF-8 character, or NULL at string end.
 */
static const char* utf8_next_codepoint(const char *str, uint32_t *codepoint) {
    if (!str || !*str) {
        *codepoint = 0;
        return NULL;
    }
    const unsigned char *s = (const unsigned char *)str;
    if (s[0] < 0x80) {
        *codepoint = s[0];
        return str + 1;
    } else if ((s[0] & 0xE0) == 0xC0) {
        if ((s[1] & 0xC0) != 0x80) {
            *codepoint = s[0];
            return str + 1; /* Malformed sequence fallback */
        }
        *codepoint = ((s[0] & 0x1F) << 6) | (s[1] & 0x3F);
        return str + 2;
    } else if ((s[0] & 0xF0) == 0xE0) {
        if ((s[1] & 0xC0) != 0x80 || (s[2] & 0xC0) != 0x80) {
            *codepoint = s[0];
            return str + 1;
        }
        *codepoint = ((s[0] & 0x0F) << 12) | ((s[1] & 0x3F) << 6) | (s[2] & 0x3F);
        return str + 3;
    } else if ((s[0] & 0xF8) == 0xF0) {
        if ((s[1] & 0xC0) != 0x80 || (s[2] & 0xC0) != 0x80 || (s[3] & 0xC0) != 0x80) {
            *codepoint = s[0];
            return str + 1;
        }
        *codepoint = ((s[0] & 0x07) << 18) | ((s[1] & 0x3F) << 12) | ((s[2] & 0x3F) << 6) | (s[3] & 0x3F);
        return str + 4;
    } else {
        *codepoint = s[0];
        return str + 1;
    }
}

/**
 * @brief Encodes a 32-bit Unicode codepoint into UTF-8 bytes.
 *
 * @param cp Unicode codepoint.
 * @param out Destination buffer (must have at least 4 bytes available).
 * @return Number of bytes written (1 to 4).
 */
static int utf8_encode_codepoint(uint32_t cp, char *out) {
    if (cp < 0x80) {
        out[0] = (char)cp;
        return 1;
    } else if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    } else if (cp < 0x10000) {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    } else if (cp < 0x110000) {
        out[0] = (char)(0xF0 | (cp >> 18));
        out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
        out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[3] = (char)(0x80 | (cp & 0x3F));
        return 4;
    }
    return 0;
}

/* ============================================================================
 * Memory Management
 * ============================================================================ */

/**
 * @brief Recursively frees a TrieNode and all its descendants.
 */
static void free_trie_node(TrieNode *node) {
    if (!node) return;

    if (node->ascii_children) {
        for (int i = 0; i < 256; i++) {
            if (node->ascii_children[i]) {
                free_trie_node(node->ascii_children[i]);
            }
        }
        free(node->ascii_children);
        node->ascii_children = NULL;
    }

    UnicodeNode *curr_u = node->unicode_children;
    while (curr_u) {
        UnicodeNode *next_u = curr_u->next;
        free_trie_node(curr_u->node);
        free(curr_u);
        curr_u = next_u;
    }
    node->unicode_children = NULL;

    free(node);
}

/**
 * @brief Frees a Trie and all its allocated nodes.
 */
static void trie_free(Trie *trie) {
    if (!trie) return;
    if (trie->root) {
        free_trie_node(trie->root);
        trie->root = NULL;
    }
    free(trie);
}

/**
 * @brief Allocates and initializes a new Trie.
 * @return Pointer to new Trie, or NULL on allocation failure.
 */
static Trie* trie_create(void) {
    Trie *trie = (Trie *)malloc(sizeof(Trie));
    if (!trie) return NULL;

    trie->root = (TrieNode *)calloc(1, sizeof(TrieNode));
    if (!trie->root) {
        free(trie);
        return NULL;
    }
    trie->word_count = 0;
    return trie;
}

/* ============================================================================
 * Trie Operations
 * ============================================================================ */

/**
 * @brief Inserts a UTF-8 word into the Trie.
 *
 * @param trie Pointer to Trie.
 * @param word Null-terminated UTF-8 string.
 * @return 1 if newly inserted, 0 if already existed, -1 on memory allocation error.
 */
static int trie_insert_internal(Trie *trie, const char *word) {
    if (!trie || !trie->root || !word || !*word) return 0;

    TrieNode *curr = trie->root;
    const char *p = word;
    uint32_t cp = 0;

    while ((p = utf8_next_codepoint(p, &cp)) != NULL) {
        TrieNode *next_node = NULL;

        if (cp < 256) {
            /* ASCII or Latin-1 character (including standard apostrophe ' = 39) */
            if (!curr->ascii_children) {
                curr->ascii_children = (TrieNode **)calloc(256, sizeof(TrieNode *));
                if (!curr->ascii_children) return -1;
            }
            next_node = curr->ascii_children[cp];
            if (!next_node) {
                next_node = (TrieNode *)calloc(1, sizeof(TrieNode));
                if (!next_node) return -1;
                curr->ascii_children[cp] = next_node;
            }
        } else {
            /* Unicode character (Cyrillic, modifier apostrophe, etc.) */
            UnicodeNode *u = curr->unicode_children;
            while (u) {
                if (u->codepoint == cp) {
                    next_node = u->node;
                    break;
                }
                u = u->next;
            }
            if (!next_node) {
                next_node = (TrieNode *)calloc(1, sizeof(TrieNode));
                if (!next_node) return -1;

                UnicodeNode *new_u = (UnicodeNode *)malloc(sizeof(UnicodeNode));
                if (!new_u) {
                    free(next_node);
                    return -1;
                }
                new_u->codepoint = cp;
                new_u->node = next_node;
                new_u->next = curr->unicode_children;
                curr->unicode_children = new_u;
            }
        }
        curr = next_node;
    }

    int is_new = 0;
    if (curr != trie->root) {
        if (!curr->is_end) {
            curr->is_end = 1;
            trie->word_count++;
            is_new = 1;
        }
    }
    return is_new;
}

/**
 * @brief Searches for a UTF-8 word in the Trie.
 *
 * @param trie Pointer to Trie.
 * @param word Null-terminated UTF-8 string.
 * @return 1 if found and marked as end of word, 0 otherwise.
 */
static int trie_search_internal(const Trie *trie, const char *word) {
    if (!trie || !trie->root || !word || !*word) return 0;

    const TrieNode *curr = trie->root;
    const char *p = word;
    uint32_t cp = 0;

    while ((p = utf8_next_codepoint(p, &cp)) != NULL) {
        if (cp < 256) {
            if (!curr->ascii_children) return 0;
            curr = curr->ascii_children[cp];
        } else {
            const UnicodeNode *u = curr->unicode_children;
            const TrieNode *found = NULL;
            while (u) {
                if (u->codepoint == cp) {
                    found = u->node;
                    break;
                }
                u = u->next;
            }
            curr = found;
        }
        if (!curr) return 0;
    }

    return curr->is_end ? 1 : 0;
}

/**
 * @brief Checks if a TrieNode has any child nodes.
 */
static int trie_node_has_children(const TrieNode *node) {
    if (!node) return 0;
    if (node->ascii_children) {
        for (int i = 0; i < 256; i++) {
            if (node->ascii_children[i]) return 1;
        }
    }
    if (node->unicode_children) return 1;
    return 0;
}

/**
 * @brief Recursively collects word completions branching from a node.
 *
 * @param node Current TrieNode in traversal.
 * @param buf Character buffer holding the current suffix string.
 * @param depth Current length in bytes in buf.
 * @param max_depth Maximum capacity of buf.
 * @param list Python list to append completion strings to.
 * @return 0 on success, -1 on error.
 */
static int collect_completions(const TrieNode *node, char *buf, int depth, int max_depth, PyObject *list) {
    if (!node) return 0;

    if (depth > 0 && node->is_end) {
        PyObject *py_str = PyUnicode_FromStringAndSize(buf, depth);
        if (!py_str) return -1;

        int contains = PySequence_Contains(list, py_str);
        if (contains == 0) {
            if (PyList_Append(list, py_str) < 0) {
                Py_DECREF(py_str);
                return -1;
            }
        } else if (contains < 0) {
            Py_DECREF(py_str);
            return -1;
        }
        Py_DECREF(py_str);
    }

    if (depth >= max_depth - 8) {
        return 0; /* Guard against buffer overflow */
    }

    /* Traverse ASCII children */
    if (node->ascii_children) {
        for (int c = 0; c < 256; c++) {
            const TrieNode *child = node->ascii_children[c];
            if (child) {
                buf[depth] = (char)c;
                if (collect_completions(child, buf, depth + 1, max_depth, list) < 0) {
                    return -1;
                }
            }
        }
    }

    /* Traverse Unicode children */
    const UnicodeNode *u = node->unicode_children;
    while (u) {
        if (u->node) {
            int enc_len = utf8_encode_codepoint(u->codepoint, buf + depth);
            if (enc_len > 0) {
                if (collect_completions(u->node, buf, depth + enc_len, max_depth, list) < 0) {
                    return -1;
                }
            }
        }
        u = u->next;
    }

    return 0;
}

/* ============================================================================
 * Python Capsule & Argument Helpers
 * ============================================================================ */

/**
 * @brief Capsule destructor called when Python garbage-collects the capsule.
 */
static void trie_capsule_destructor(PyObject *capsule) {
    Trie *trie = (Trie *)PyCapsule_GetPointer(capsule, TRIE_CAPSULE_NAME);
    if (trie) {
        trie_free(trie);
        PyCapsule_SetPointer(capsule, NULL);
    } else {
        PyErr_Clear();
    }
}

/**
 * @brief Extracts the Trie pointer from a Python capsule object with validation.
 */
static Trie* get_trie_from_capsule(PyObject *capsule) {
    if (!capsule || !PyCapsule_CheckExact(capsule)) {
        PyErr_SetString(PyExc_TypeError, "First argument must be a valid Trie capsule object");
        return NULL;
    }
    Trie *trie = (Trie *)PyCapsule_GetPointer(capsule, TRIE_CAPSULE_NAME);
    if (!trie) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_ValueError, "Trie capsule is invalid or has already been freed");
        }
        return NULL;
    }
    return trie;
}

/* ============================================================================
 * Python Module Functions
 * ============================================================================ */

/**
 * @brief Python: create_trie() -> capsule object
 */
static PyObject* py_create_trie(PyObject *self, PyObject *Py_UNUSED(args)) {
    Trie *trie = trie_create();
    if (!trie) {
        PyErr_NoMemory();
        return NULL;
    }

    PyObject *capsule = PyCapsule_New((void *)trie, TRIE_CAPSULE_NAME, trie_capsule_destructor);
    if (!capsule) {
        trie_free(trie);
        return NULL;
    }

    return capsule;
}

/**
 * @brief Python: trie_insert(trie, word: str) -> None
 */
static PyObject* py_trie_insert(PyObject *self, PyObject *args) {
    PyObject *capsule = NULL;
    const char *word = NULL;

    if (!PyArg_ParseTuple(args, "Os:trie_insert", &capsule, &word)) {
        return NULL;
    }

    Trie *trie = get_trie_from_capsule(capsule);
    if (!trie) return NULL;

    int res = trie_insert_internal(trie, word);
    if (res < 0) {
        PyErr_NoMemory();
        return NULL;
    }

    Py_RETURN_NONE;
}

/**
 * @brief Python: trie_search(trie, word: str) -> bool
 */
static PyObject* py_trie_search(PyObject *self, PyObject *args) {
    PyObject *capsule = NULL;
    const char *word = NULL;

    if (!PyArg_ParseTuple(args, "Os:trie_search", &capsule, &word)) {
        return NULL;
    }

    Trie *trie = get_trie_from_capsule(capsule);
    if (!trie) return NULL;

    int found = trie_search_internal(trie, word);
    if (found) {
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
}

/**
 * @brief Python: trie_bulk_insert(trie, words: list) -> int
 */
static PyObject* py_trie_bulk_insert(PyObject *self, PyObject *args) {
    PyObject *capsule = NULL;
    PyObject *words_obj = NULL;

    if (!PyArg_ParseTuple(args, "OO:trie_bulk_insert", &capsule, &words_obj)) {
        return NULL;
    }

    Trie *trie = get_trie_from_capsule(capsule);
    if (!trie) return NULL;

    PyObject *fast_seq = PySequence_Fast(words_obj, "Argument 'words' must be an iterable/sequence of strings");
    if (!fast_seq) return NULL;

    Py_ssize_t n = PySequence_Fast_GET_SIZE(fast_seq);
    int count = 0;

    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(fast_seq, i);
        if (!PyUnicode_Check(item)) {
            Py_DECREF(fast_seq);
            PyErr_Format(PyExc_TypeError, "Item at index %zd in words list must be a str, got %.200s",
                         i, Py_TYPE(item)->tp_name);
            return NULL;
        }

        const char *utf8_str = PyUnicode_AsUTF8(item);
        if (!utf8_str) {
            Py_DECREF(fast_seq);
            return NULL;
        }

        int res = trie_insert_internal(trie, utf8_str);
        if (res < 0) {
            Py_DECREF(fast_seq);
            PyErr_NoMemory();
            return NULL;
        }
        if (res > 0) {
            count++;
        }
    }

    Py_DECREF(fast_seq);
    return PyLong_FromLong(count);
}

/**
 * @brief Python: trie_find_suffixes(trie, word: str) -> list of matching suffix strings
 *
 * Performs suffix matching for Uzbek morphological analysis:
 * 1. Checks all suffix slices of `word` (from longest to shortest) that exist
 *    in the Trie (affix dictionary lookup).
 * 2. Checks if `word` exists as a prefix in the Trie, collecting descendant
 *    completions (stem completion / autocomplete).
 */
static PyObject* py_trie_find_suffixes(PyObject *self, PyObject *args) {
    PyObject *capsule = NULL;
    const char *word = NULL;

    if (!PyArg_ParseTuple(args, "Os:trie_find_suffixes", &capsule, &word)) {
        return NULL;
    }

    Trie *trie = get_trie_from_capsule(capsule);
    if (!trie) return NULL;

    PyObject *result_list = PyList_New(0);
    if (!result_list) return NULL;

    if (!word || !*word) {
        return result_list;
    }

    /* Check whether `word` exists as a prefix and has descendant completions */
    const TrieNode *word_node = trie->root;
    int prefix_exists = 1;
    const char *p_pref = word;
    uint32_t cp = 0;

    while ((p_pref = utf8_next_codepoint(p_pref, &cp)) != NULL) {
        if (cp < 256) {
            if (!word_node->ascii_children || !word_node->ascii_children[cp]) {
                prefix_exists = 0;
                break;
            }
            word_node = word_node->ascii_children[cp];
        } else {
            const UnicodeNode *u = word_node->unicode_children;
            const TrieNode *found = NULL;
            while (u) {
                if (u->codepoint == cp) {
                    found = u->node;
                    break;
                }
                u = u->next;
            }
            if (!found) {
                prefix_exists = 0;
                break;
            }
            word_node = found;
        }
    }

    int word_has_descendants = (prefix_exists && word_node && trie_node_has_children(word_node));

    /* Step 1: Scan all suffixes of `word` across UTF-8 character boundaries */
    size_t word_len = strlen(word);
    const char *curr_ptr = word;

    while (curr_ptr < word + word_len) {
        int is_full_word = (curr_ptr == word);

        /* Only include the full word if it does not have descendant completions */
        if (!is_full_word || !word_has_descendants) {
            if (trie_search_internal(trie, curr_ptr)) {
                PyObject *py_str = PyUnicode_FromString(curr_ptr);
                if (!py_str) {
                    Py_DECREF(result_list);
                    return NULL;
                }
                int contains = PySequence_Contains(result_list, py_str);
                if (contains == 0) {
                    if (PyList_Append(result_list, py_str) < 0) {
                        Py_DECREF(py_str);
                        Py_DECREF(result_list);
                        return NULL;
                    }
                } else if (contains < 0) {
                    Py_DECREF(py_str);
                    Py_DECREF(result_list);
                    return NULL;
                }
                Py_DECREF(py_str);
            }
        }

        curr_ptr = utf8_next_codepoint(curr_ptr, &cp);
        if (!curr_ptr) break;
    }

    /* Step 2: If `word` has completions in the Trie, collect remaining suffix completions */
    if (prefix_exists && word_node && word_has_descendants) {
        char buf[MAX_COMPLETION_DEPTH];
        if (collect_completions(word_node, buf, 0, sizeof(buf), result_list) < 0) {
            Py_DECREF(result_list);
            return NULL;
        }
    }

    return result_list;
}

/**
 * @brief Python: free_trie(trie) -> None
 */
static PyObject* py_free_trie(PyObject *self, PyObject *args) {
    PyObject *capsule = NULL;

    if (!PyArg_ParseTuple(args, "O:free_trie", &capsule)) {
        return NULL;
    }

    if (!PyCapsule_CheckExact(capsule)) {
        PyErr_SetString(PyExc_TypeError, "First argument must be a valid Trie capsule object");
        return NULL;
    }

    Trie *trie = (Trie *)PyCapsule_GetPointer(capsule, TRIE_CAPSULE_NAME);
    if (!trie) {
        /* Already freed or invalid capsule; clear error and return None gracefully */
        PyErr_Clear();
        Py_RETURN_NONE;
    }

    trie_free(trie);
    PyCapsule_SetPointer(capsule, NULL);
    PyCapsule_SetName(capsule, NULL);
    PyCapsule_SetDestructor(capsule, NULL);

    Py_RETURN_NONE;
}

/* ============================================================================
 * Module Initialization
 * ============================================================================ */

static PyMethodDef TrieMethods[] = {
    {
        "create_trie",
        (PyCFunction)py_create_trie,
        METH_NOARGS,
        "create_trie() -> PyCapsule\n"
        "Create and return a new Trie capsule object."
    },
    {
        "trie_insert",
        (PyCFunction)py_trie_insert,
        METH_VARARGS,
        "trie_insert(trie, word: str) -> None\n"
        "Insert a UTF-8 word into the Trie."
    },
    {
        "trie_search",
        (PyCFunction)py_trie_search,
        METH_VARARGS,
        "trie_search(trie, word: str) -> bool\n"
        "Search for a UTF-8 word in the Trie. Returns True if found, False otherwise."
    },
    {
        "trie_bulk_insert",
        (PyCFunction)py_trie_bulk_insert,
        METH_VARARGS,
        "trie_bulk_insert(trie, words: list) -> int\n"
        "Bulk insert a list/iterable of words into the Trie. Returns count of words inserted."
    },
    {
        "trie_find_suffixes",
        (PyCFunction)py_trie_find_suffixes,
        METH_VARARGS,
        "trie_find_suffixes(trie, word: str) -> list\n"
        "Find matching suffix strings for a given word."
    },
    {
        "free_trie",
        (PyCFunction)py_free_trie,
        METH_VARARGS,
        "free_trie(trie) -> None\n"
        "Explicitly free the Trie capsule object and release all memory."
    },
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static struct PyModuleDef ctrie_module = {
    PyModuleDef_HEAD_INIT,
    "_ctrie",
    "Fast Trie implementation in C for uztoken.",
    -1,
    TrieMethods
};

PyMODINIT_FUNC PyInit__ctrie(void) {
    return PyModule_Create(&ctrie_module);
}
