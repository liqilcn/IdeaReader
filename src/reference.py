import MySQLdb
import MySQLdb.cursors
import pymysql

# In[36]:


def contruct_ref_gbt(paper_details):
    # Bazzi M, Porter M A, Williams S, et al. Community detection in temporal multilayer networks, with an application to correlation networks[J]. Multiscale Modeling & Simulation, 2016, 14(1): 1-41.
    # He J, Chen D. A fast algorithm for community detection in temporal network[J]. Physica A: Statistical Mechanics and its Applications, 2015, 429: 87-94.
    # Jia Y, Zhang Q, Zhang W, et al. CommunityGAN: Community detection with generative adversarial nets[C]//The World Wide Web Conference. 2019: 784-794.
    # Wang H, Wang J, Wang J, et al. Graphgan: Graph representation learning with generative adversarial nets[C]//Thirty-second AAAI conference on artificial intelligence. 2018.

    anames = []
    for aname in paper_details['names']:
        aname = aname.split()
        aname = ' '.join([aname[-1], *(i[0] for i in aname[:-1])])
        anames.append(aname)
    if len(paper_details['names']) > 3:
        anames.append('et al')
    author = ', '.join(anames)

    # 论文类型 0未设置 1Book 2BookChapter 3Conference 4Journal 5Patent 6Dataset 7Repository 8Thesis
    venue_type = ['Z', 'M', 'M', 'C', 'J', 'P', 'DS', 'A', 'D'][paper_details['doc_type']]

    volume_issue = ''
    if paper_details['volume'] and paper_details['issue']:
        volume_issue = ', %d(%d)' % (paper_details['volume'], paper_details['issue'])
    elif paper_details['volume']:
        volume_issue = ', %d' % paper_details['volume']

    pages = ''
    if paper_details['first_page'] and paper_details['last_page']:
        pages = ': %d-%d' % (paper_details['first_page'], paper_details['last_page'])
    title = paper_details['title']
    venue_name = paper_details['venue_name']
    year = paper_details['year']
    if paper_details['doc_type'] == 3:
        cite = f'{author}. {title}[{venue_type}]//{venue_name}. {year}{volume_issue}{pages}.'
    else:
        cite = f'{author}. {title}[{venue_type}]. {venue_name}, {year}{volume_issue}{pages}.'
    return cite


# In[37]:


def get_papers_formated_reference(pid_list):
    # 输入是需要处理的paper_id列表，输出是pid与按GB/T 7714格式化的reference字符串
    db = MySQLdb.connect(
        host = 'xxx',
        user = 'xxx',
        password = 'xxx',
        db = 'xxx',
        port = 80,
        charset = 'utf8mb4',
        cursorclass=MySQLdb.cursors.SSCursor
    )
    cursor = db.cursor()
    pid2ref = {}
    for p_id in pid_list:
        paper_details = {}
        sql = f"SELECT title, year, doc_type, journal_id, conference_series_id, volume, issue, first_page, last_page FROM `am_paper`.`am_paper` WHERE paper_id = {p_id}"
        cursor.execute(sql)
        data = cursor.fetchone()
        paper_details['title'] = data[0]
        paper_details['year'] = data[1]
        paper_details['doc_type'] = data[2]
        paper_details['volume'] = data[5]
        paper_details['issue'] = data[6]
        paper_details['first_page'] = data[7]
        paper_details['last_page'] = data[8]
        if data[3] != 0:
            sql = f"SELECT name FROM `am_paper`.`am_journal` WHERE journal_id = {data[3]}"
            cursor.execute(sql)
            j_name = cursor.fetchone()
            paper_details['venue_name'] = j_name[0]
        elif data[4] != 0:
            sql = f"SELECT name FROM `am_paper`.`am_conference_series` WHERE conference_series_id = {data[4]}"
            cursor.execute(sql)
            c_name = cursor.fetchone()
            paper_details['venue_name'] = c_name[0]
        else:
            paper_details['venue_name'] = ''
        sql = f"SELECT author_id FROM `am_paper`.`am_paper_author` WHERE paper_id = {p_id}"
        cursor.execute(sql)
        aids = cursor.fetchall()
        anames = []
        for a_id in aids:
            sql = f"SELECT name FROM `am_paper`.`am_author` WHERE author_id = {a_id[0]}"
            cursor.execute(sql)
            a_name = cursor.fetchone()
            anames.append(a_name[0])
        paper_details['names'] = anames
        ref = contruct_ref_gbt(paper_details)
        pid2ref[p_id] = ref
    db.close()
    return pid2ref


# In[40]:


if __name__ == "__main__":
    results = get_papers_formated_reference(['118963114','423634839'])
    print(results)

