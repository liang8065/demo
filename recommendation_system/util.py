#coding: utf-8
import json
import sys
import cPickle as pickle

def parse_song_line(in_line):
    data = json.loads(in_line)
    name = data['result']['name']
    tags = ",".join(data['result']['tags'])
    subscribed_count = data['result']['subscribedCount']
    if(subscribed_count < 100):
        return False
    playlist_id = data['result']['id']
    song_info = ''
    songs = data['result']['tracks']
    for song in songs:
        try:
            song_info += "\t" + ":::".join([str(song['id']), song['name'], song['artists'][0]['name'], str(song['popularity'])])
        except Exception, e:
            continue
    return name + "##" + tags + "##" + str(playlist_id) + "##" + str(subscribed_count) + song_info

def parse_json_file(in_file, out_file):
    out = open(out_file, 'w')
    for line in open(in_file):
        result = parse_song_line(line)
        if(result):
            out.write(result.encode('utf-8').strip()+"\n")
    out.close()

def is_null(s):
    return len(s.split(",")) > 2

def parse_song_info(song_info):
    try:
        song_id, name, artist, popularity = song_info.split(":::")
        return ",".join([song_id, "1.0"])
    except Exception,e:
        return "";

def parse_playlist_line(in_line):
    try:
        contents = in_line.strip().split("\t")
        name, tags, playlist_id, subscribed_count = contents[0].split("##")
        songs_info = map(lambda x : playlist_id + "," + parse_song_info(x), contents[1:])
        songs_info = filter(is_null, songs_info)
        return "\n".join(songs_info)
    except Exception, e:
        return False

def parse_list_file(in_file, out_file):
    out = open(out_file, 'w')
    for line in open(in_file):
        result = parse_playlist_line(line)
        if(result):
            out.write(result.encode('utf-8').strip()+"\n")
    out.close()

def parse_playlist_get_info(in_line, playlist_dic, song_dic):
    contents = in_line.strip().split("\t")
    name, tags, playlist_id, subscribed_count = contents[0].split("##")
    playlist_dic[playlist_id] = name
    for song in contents[1:]:
        try:
            song_id, song_name, artist, popularity = song.split(":::")
            song_dic[song_id] = song_name+"\t"+artist
        except:
            print "song format error"
            print song+"\n"

def parse_file(in_file, out_playlist, out_song):
    playlist_dic = {}
    song_dic = {}
    for line in open(in_file):
        parse_playlist_get_info(line, playlist_dic, song_dic)
    pickle.dump(playlist_dic, open(out_playlist,"wb"))
    pickle.dump(song_dic, open(out_song,"wb"))

if __name__ == "__main__":
    #parse_json_file("../../data/recommendation_system/playlistdetail.all.json", "../../data/recommendation_system/163_music_playlist.txt")
    #parse_list_file("../../data/recommendation_system/163_music_playlist.txt", "../../data/recommendation_system/163_music_suprise_format.txt")
    #parse_list_file("../../data/recommendation_system/popular.playlist", "../../data/recommendation_system/popular_music_suprise_format.txt")
    #parse_file("../../data/recommendation_system/163_music_playlist.txt",
    #        "../../data/recommendation_system/playlist.pkl", "../../data/recommendation_system/song.pkl")
    parse_file("../../data/recommendation_system/popular.playlist",
            "../../data/recommendation_system/popular_playlist.pkl",
            "../../data/recommendation_system/popular_song.pkl")

