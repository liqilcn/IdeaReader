from typing import Optional
import mysql.connector


def getReadDB(database: Optional[str]):
    if str is None:
        return mysql.connector.connect(
            host="xxxx",
            user="xxxx",  # remote
            password="xxxx",  # bJuPOIQn9LuNZqmFR9qa
            port=3306
        )
    else:
        return mysql.connector.connect(
            host="xxxx",
            user="xxxx",
            password="xxxx",
            port=3306,
            database=database
        )


def acemap_sort_downloader(pid: str) -> Optional[dict]:
    db = getReadDB("am_paper")
    cursor = db.cursor()

    ret = {'_id': pid,
           'id': pid,
           'title': '',
           'abstract': '',
           'citations': [],  # the pids of citation papers
           'references': [],  # the pids of reference papers
           'year': 0
           }

    cursor.execute(
        "SELECT title, year, journal_id, conference_instance_id FROM am_paper WHERE paper_id = %(paper_id)s;",
        {'paper_id': int(pid)})
    data = cursor.fetchone()
    if data is None:
        cursor.close()
        db.close()
        return None
    ret['title'] = data[0]
    ret['year'] = data[1]

    cursor.execute("SELECT abstract FROM am_paper_abstract WHERE paper_id = %(paper_id)s;", {'paper_id': int(pid)})
    d = cursor.fetchone()
    if d is not None:
        ret['abstract'] = d[0]

    cursor.execute("SELECT reference_id FROM am_paper_reference WHERE paper_id = %(paper_id)s LIMIT 0, 1000;",
                   {'paper_id': int(pid)})
    d = cursor.fetchall()
    if d is not None:
        ret['references'] = [str(t[0]) for t in d]

    cursor.execute('''SELECT paper_id AS pid FROM am_paper_reference WHERE reference_id = %(reference_id)s ORDER BY (
        SELECT COUNT(*) FROM am_paper_reference WHERE reference_id = pid
        ) DESC LIMIT 0, 1000;''', {'reference_id': int(pid)})
    d = cursor.fetchall()
    if d is not None:
        ret['citations'] = [str(t[0]) for t in d]

    cursor.close()
    db.close()
    return ret


def acemap_unsort_downloader(pid: str) -> Optional[dict]:
    db = getReadDB("am_paper")
    cursor = db.cursor()

    ret = {'_id': pid,
           'id': pid,
           'title': '',
           'abstract': '',
           'citations': [],  # the pids of citation papers
           'references': [],  # the pids of reference papers
           'year': 0
           }

    cursor.execute(
        "SELECT title, year, journal_id, conference_instance_id FROM am_paper WHERE paper_id = %(paper_id)s;",
        {'paper_id': int(pid)})
    data = cursor.fetchone()
    if data is None:
        cursor.close()
        db.close()
        return None
    ret['title'] = data[0]
    ret['year'] = data[1]

    cursor.execute("SELECT abstract FROM am_paper_abstract WHERE paper_id = %(paper_id)s;", {'paper_id': int(pid)})
    d = cursor.fetchone()
    if d is not None:
        ret['abstract'] = d[0]

    cursor.execute("SELECT reference_id FROM am_paper_reference WHERE paper_id = %(paper_id)s LIMIT 0, 1000;",
                   {'paper_id': int(pid)})
    d = cursor.fetchall()
    if d is not None:
        ret['references'] = [str(t[0]) for t in d]

    cursor.execute('''SELECT paper_id AS pid FROM am_paper_reference WHERE reference_id = %(reference_id)s 
                        LIMIT 0, 1000;''', {'reference_id': int(pid)})
    d = cursor.fetchall()
    if d is not None:
        ret['citations'] = [str(t[0]) for t in d]

    cursor.close()
    db.close()
    return ret


def getReferenceCount(pid: str) -> int:
    db = getReadDB("am_paper")
    cursor = db.cursor()
    cursor.execute("SELECT reference_id FROM am_paper_reference WHERE paper_id = %(paper_id)s LIMIT 0, 1000;",
                   {'paper_id': int(pid)})
    d = cursor.fetchall()
    cursor.close()
    db.close()
    return len(d)


def getCitationCount(pid: str) -> int:
    db = getReadDB("am_paper")
    cursor = db.cursor()
    cursor.execute("SELECT paper_id FROM am_paper_reference WHERE reference_id = %(paper_id)s LIMIT 0, 1000;",
                   {'paper_id': int(pid)})
    d = cursor.fetchall()
    cursor.close()
    db.close()
    return len(d)


def getPaperInformation(pid: str) -> Optional[dict]:
    db = getReadDB("am_paper")
    cursor = db.cursor()

    ret = {
        "paper_title": "",
        "paper_authors": [],
        "paper_year": 0,
        "paper_venue": "",
        "paper_abstract": ""
    }

    cursor.execute(
        "SELECT title, year, journal_id, conference_instance_id FROM am_paper WHERE paper_id = %(paper_id)s;",
        {'paper_id': int(pid)})
    data = cursor.fetchone()
    if data is None:
        cursor.close()
        db.close()
        return None
    ret['paper_title'] = data[0]
    ret['paper_year'] = data[1]
    if data[2] != 0:  # journal
        cursor.execute("SELECT name FROM am_journal WHERE journal_id = %(journal_id)s;", {'journal_id': data[2]})
        d = cursor.fetchone()
        if d is not None:
            ret['paper_venue'] = d[0]
    elif data[3] != 0:  # conference
        cursor.execute("SELECT name FROM am_conference_series WHERE conference_series_id = %(conference_series_id)s;",
                       {'conference_series_id': data[3]})
        d = cursor.fetchone()
        if d is not None:
            ret['paper_venue'] = d[0]

    cursor.execute("SELECT abstract FROM am_paper_abstract WHERE paper_id = %(paper_id)s;", {'paper_id': int(pid)})
    d = cursor.fetchone()
    if d is not None:
        ret['paper_abstract'] = d[0]

    cursor.execute('''SELECT name FROM am_author WHERE author_id in
                        (SELECT author_id FROM am_paper_author WHERE paper_id = %(paper_id)s)
                      ;''', {'paper_id': int(pid)})
    for row in cursor:
        ret['paper_authors'].append(row[0])

    cursor.close()
    db.close()
    return ret


def getCompactTree(pid: str, tree_type: str) -> Optional[str]:
    db = getReadDB("skeleton_based_sum")
    cursor = db.cursor()

    cursor.execute(
        "SELECT `mrt_json` FROM `mrt_tree` WHERE `field_id` = %(paper_id)s AND `type` = %(type)s;",
        {'paper_id': int(pid), 'type': tree_type})
    data = cursor.fetchone()
    if data is None:
        cursor.close()
        db.close()
        return None
    else:
        cursor.close()
        db.close()
        return data[0]
