

CREATE DATABASE IF NOT EXISTS myflaskapp;

Use myflaskapp;

CREATE TABLE IF NOT EXISTS `users` (
  `uid` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(100) DEFAULT NULL,
  `email` varchar(100) DEFAULT NULL,
  `username` varchar(30) DEFAULT NULL,
  `password` varchar(100) DEFAULT NULL,
  `register_date` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`uid`)
);

CREATE TABLE IF NOT EXISTS `movies` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `title` varchar(255) DEFAULT NULL,  
  `genres` text DEFAULT NULL,
  `cast` text DEFAULT NULL,
  `crew` varchar(255) DEFAULT NULL,
  `runtime` int(11) DEFAULT NULL, 
  `vote_count` int(11) DEFAULT 0, 
  `monthFact` int(11) DEFAULT 0,
  `actorFact` int(11) DEFAULT 0,
  `genFact` int(11) DEFAULT 0,
  `budget` decimal(20,2) DEFAULT 0,
  `revenue` decimal(20,2) DEFAULT 0,     
  `popularity` decimal(5,2) DEFAULT 0,
  `vote_average` decimal(5,2) DEFAULT 0,
  `IMDB` decimal(3,2) DEFAULT 0,
  `rotten` decimal(3,2) DEFAULT 0,
  `metaC` decimal(3,2) DEFAULT 0,
  `release_date` date DEFAULT NULL,
  `created` datetime DEFAULT CURRENT_TIMESTAMP,  
    PRIMARY KEY (`id`)
);
    

CREATE TABLE IF NOT EXISTS `predict` (
  `inpid` int(11) NOT NULL AUTO_INCREMENT,
  `budget` decimal(20,2) DEFAULT 0,
  `genres` varchar(35) DEFAULT NULL,
  `popularity` decimal(5,2) DEFAULT 0,
  `vote_count` int(11) DEFAULT 0,   
  `cast` varchar(50) DEFAULT NULL,
  `release_date` date DEFAULT NULL,  
  `uid` int(11) DEFAULT NULL,
  `revenue` decimal(20,2) DEFAULT 0,
  `typeReg` varchar(2) DEFAULT NULL,     
  PRIMARY KEY (`inpid`)
);

LOAD DATA INFILE '/var/lib/mysql-files/movies.csv' 
INTO TABLE movies 
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
IGNORE 1 LINES
(`title`,`genres`,`cast`,`crew`,`runtime`,`vote_count`,`monthFact`,`actorFact`,`genFact`,`budget`,`revenue`,`popularity`,`vote_average`,`IMDB`,`rotten`,`metaC`,`release_date`);

UPDATE movies set created='1980-06-05';
