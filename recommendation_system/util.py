#coding: utf-8
import pandas as pd

def expand_id_column(df, src_col, dst_col):
    id_list = list()
    i = 1
    tmp_dic = {}
    for item in df[src_col]:
        if item not in tmp_dic:
            tmp_dic[item] = i
            i += 1
        id_list.append(tmp_dic[item])

    df[dst_col] = pd.Series(id_list)


def parse_file(in_file, in_file_with_id, surprise_format, out_user, out_movie):
    df = pd.read_csv(in_file)

    #expand_id_column(df, "user_id", "i_user_id")
    expand_id_column(df, "movie", "movie_id")

    df.to_csv(in_file_with_id, index = False)
    df[["user_id", "movie_id", "rating"]].to_csv(surprise_format, header = False, index = False)
    df[["user_id", "user"]].drop_duplicates(subset='user_id', keep='first', inplace=False) \
                    .to_csv(out_user, header = False, index = False)
    df[["movie_id", "movie"]].drop_duplicates(subset='movie_id', keep='first', inplace=False) \
                    .to_csv(out_movie, header = False, index = False)


if __name__ == "__main__":
    parse_file("../../data/recommendation_system/douban/douban_movie.txt",
            "../../data/recommendation_system/douban/douban_movie_with_id.txt",
            "../../data/recommendation_system/douban/suprise_format.txt",
            "../../data/recommendation_system/douban/user.pkl",
            "../../data/recommendation_system/douban/movie.pkl")

