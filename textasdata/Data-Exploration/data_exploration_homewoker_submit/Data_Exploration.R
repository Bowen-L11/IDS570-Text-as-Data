# ============================================================
# IDS 570 Data Exploration -- Freedom option
# Alliances, incumbent survival, and foreign aid:
# vocabulary and syntax in political science articles,
# with a separate exploration of three official speeches.
# ============================================================
# Question: How do these three literatures differ in vocabulary,
# where do they overlap, and what does syntax add to the comparison?
# Speech extension: How do sovereignty, cooperation and security
# language appear in context, and how does student coding qualify it?

# ---- 0. Setup ----
# First open Data_Exploration.Rproj in this folder, then open this script.
# Run from the top, or click Source to run all sections.
# Install these packages once if needed:
# install.packages(c("quanteda", "quanteda.textstats", "udpipe",
#                    "dplyr", "tidyr", "readr", "ggplot2"))
library(quanteda)
library(quanteda.textstats)
library(udpipe)
library(dplyr)
library(tidyr)
library(readr)
library(ggplot2)

# The RStudio project makes this folder the working directory.
# If opened without the project, select this folder's metadata.csv.
if (!file.exists("metadata.csv")) {
  if (!interactive()) stop("Run from the folder containing metadata.csv.")
  metadata_file <- file.choose()
  if (basename(metadata_file) != "metadata.csv") stop("Please select metadata.csv.")
  setwd(dirname(metadata_file))
}
root <- normalizePath(".", winslash = "/")
if (.Platform$OS.type == "windows") {
  Sys.setlocale("LC_CTYPE", "English_United States.utf8")
}
for (folder in c("results", "figures", "models")) {
  dir.create(folder, showWarnings = FALSE)
}
# Model download is only needed on the first run on a new computer.
if (!length(list.files("models", pattern = "udpipe$"))) {
  udpipe_download_model(language = "english-ewt", model_dir = "models")
}

options(width=140)
out <- function(name) file.path(root,'results',name)
fig <- function(name) file.path(root,'figures',name)
theme_set(theme_minimal(base_size=12)+theme(panel.grid.minor=element_blank(),plot.title=element_text(face='bold')))
pal <- c(A='#236B8E',B='#A04A34',C='#5E6F3D',S='#71558C')


# ---- 1. Load the corpus: 18 articles and 3 official speeches ----
# A: alliances/bandwagoning; B: incumbent survival; C: aid/conditionality.
# Speeches are analysed separately to avoid mixing genres in TF-IDF.
meta <- read_csv(file.path(root,'metadata.csv'),show_col_types=FALSE)
stopifnot(nrow(meta)==21, !anyDuplicated(meta$doc_id), all(table(meta$group)==c(6,6,6,3)))
txt <- vapply(meta$text_file,function(p)read_file(file.path(root,p)),character(1))
names(txt) <- meta$doc_id
stopifnot(all(nchar(txt)>4000))
texts <- data.frame(doc_id=meta$doc_id,text=txt)
corp <- corpus(texts,docid_field='doc_id',text_field='text')


# ---- 2. Clean texts and build the document-feature matrix ----
# Normalize long s; tokenize, lowercase, remove punctuation and numbers.
texts$text <- gsub('\u017f','s',texts$text,fixed=TRUE)
texts$text <- gsub('[\u2018\u2019]',"'",texts$text,perl=TRUE)
texts$text <- gsub("\\b([A-Za-z]+)'s\\b",'\\1',texts$text,perl=TRUE)
corp <- corpus(texts,docid_field='doc_id',text_field='text')
tok_raw <- tokens(corp,remove_punct=TRUE,remove_symbols=TRUE,remove_numbers=TRUE,
                  remove_url=TRUE,split_hyphens=TRUE) |> tokens_tolower()
# Remove English stopwords; retain alphabetic words of at least two letters.
# No stemming is applied.
# Sentence-preserving input files above remain unchanged for UDPipe.
sw <- stopwords('en',source='snowball')
# Manually audited notation/formatting artifacts, not theoretical vocabulary.
artifacts <- c('cid','ref','fig','eq','bc','diff','isq')
write_lines(artifacts,out('artifact_terms_removed.txt'))
write_lines(sw,out('stopwords_used.txt'))
sw <- c(sw,artifacts)
tok <- tokens_remove(tok_raw,pattern=sw) |>
  tokens_select(pattern="^[a-z][a-z'-]*[a-z]$",valuetype='regex',selection='keep')
token_list <- as.list(tok)
tidy_tokens <- bind_rows(lapply(names(token_list),function(id)tibble(doc_id=id,token_index=seq_along(token_list[[id]]),word=token_list[[id]])))
write_csv(tidy_tokens,out('clean_word_tokens.csv.gz'))
counts <- meta |> select(doc_id,group,genre,title) |>
 mutate(before_stopwords=as.integer(ntoken(tok_raw)),cleaned_tokens=as.integer(ntoken(tok)),types=as.integer(ntype(tok)))
write_csv(counts,out('corpus_summary.csv'))
# Build count matrices separately for articles and speeches.
all_dfm <- dfm(tok)
articles <- all_dfm[meta$genre=='journal_article',] |> dfm_trim(min_termfreq=1)
speeches <- all_dfm[meta$genre=='official_speech',] |> dfm_trim(min_termfreq=1)
saveRDS(list(metadata=meta,counts=articles,speeches=speeches),out('dfm_counts.rds'))
tidy_matrix <- function(mat,value='value') {
 ans <- as.data.frame(as.matrix(mat));ans$doc_id <- rownames(ans)
 ans |> pivot_longer(-doc_id,names_to='word',values_to=value)
}


# ---- 3. Approach 1: TF-IDF distinctive vocabulary ----
tfidf <- dfm_tfidf(articles,scheme_tf='count',scheme_df='inverse',base=10)
tfidf_long <- tidy_matrix(tfidf,'tfidf') |> filter(tfidf>0)
top15 <- tfidf_long |> group_by(doc_id) |> arrange(desc(tfidf),word,.by_group=TRUE) |> slice_head(n=15) |> ungroup()
write_csv(top15,out('tfidf_top15_articles.csv'))
write_csv(tfidf_long,out('tfidf_all_articles.csv.gz'))
shared <- top15 |> group_by(word) |> summarise(documents=n_distinct(doc_id),doc_ids=paste(sort(unique(doc_id)),collapse='; '),.groups='drop') |> filter(documents>=2) |> arrange(desc(documents),word)
write_csv(shared,out('shared_distinctive_terms.csv'))
# Normalizing TF within a document does not change its top-term ranking.
tfidf_prop <- dfm_tfidf(articles,scheme_tf='prop',base=10)
write_csv(tidy_matrix(tfidf_prop,'tfidf_proportional') |> filter(tfidf_proportional>0),out('tfidf_proportional_articles.csv.gz'))
for(g in c('A','B','C')) {
 pp <- top15 |> filter(substr(doc_id,1,1)==g) |> group_by(doc_id) |> slice_head(n=10) |> ungroup() |>
  mutate(key=paste(doc_id,word,sep='___'),key=factor(key,levels=rev(unique(key))))
 p <- ggplot(pp,aes(tfidf,key))+geom_col(fill=pal[g],width=.75)+facet_wrap(~doc_id,scales='free',ncol=3)+
  scale_y_discrete(labels=function(x)sub('.*___','',x))+
  labs(title=paste('Distinctive vocabulary \u2014 group',g),subtitle='Top 10 per article; IDF calculated across all 18 articles',x='TF-IDF (raw count \u00d7 log10[18 / document frequency])',y=NULL,
       caption='A: alliances/bandwagoning  |  B: incumbent survival  |  C: external demands and compliance')
 ggsave(fig(paste0('tfidf_group_',g,'.png')),p,width=13,height=8,dpi=180)
}


# ---- 4. Approach 2: Pearson similarity between articles ----
trimmed <- dfm_trim(articles,min_termfreq=5)
r <- as.matrix(textstat_simil(trimmed,margin='documents',method='correlation'))
stopifnot(isTRUE(all.equal(unname(r),unname(cor(t(as.matrix(trimmed)))),tolerance=1e-8)))
r[] <- pmax(-1,pmin(1,r));diag(r) <- 1 # floating-point roundoff must not create grey NA tiles
write_csv(as.data.frame(r) |> tibble::rownames_to_column('doc_id'),out('pearson_matrix.csv'))
ix <- which(upper.tri(r),arr.ind=TRUE)
pairs <- tibble(doc_i=rownames(r)[ix[,1]],doc_j=colnames(r)[ix[,2]],pearson=r[ix]) |>
 mutate(group_i=substr(doc_i,1,1),group_j=substr(doc_j,1,1),same_group=group_i==group_j) |> arrange(desc(pearson))
stopifnot(nrow(pairs)==153)
write_csv(pairs,out('pearson_all_pairs.csv'))
author_map <- setNames(strsplit(meta$authors,'; ',fixed=TRUE),meta$doc_id)
pair_groups <- pairs |> mutate(group_pair=paste(pmin(group_i,group_j),pmax(group_i,group_j),sep='\u2013'),
 shared_author=mapply(function(i,j)length(intersect(author_map[[i]],author_map[[j]]))>0,doc_i,doc_j))
write_csv(pair_groups,out('pearson_pairs_with_author_check.csv'))
summarize_groups <- function(x) x |> group_by(group_pair) |> summarise(pairs=n(),mean_r=mean(pearson),median_r=median(pearson),min_r=min(pearson),max_r=max(pearson),.groups='drop')
write_csv(summarize_groups(pair_groups),out('pearson_group_summary.csv'))
write_csv(summarize_groups(filter(pair_groups,!shared_author)),out('pearson_group_without_shared_authors.csv'))
write_csv(bind_rows(slice_head(pairs,n=2)|>mutate(rank_type='most similar'),slice_tail(pairs,n=2)|>arrange(pearson)|>mutate(rank_type='least similar')),out('pearson_extreme_pairs.csv'))
rfull <- cor(t(as.matrix(articles)))
sens <- pairs |> mutate(pearson_untrimmed=mapply(function(a,b)rfull[a,b],doc_i,doc_j),difference=pearson-pearson_untrimmed)
write_csv(sens,out('pearson_trim_sensitivity.csv'))
heat <- as.data.frame(r) |> tibble::rownames_to_column('doc_i') |> pivot_longer(-doc_i,names_to='doc_j',values_to='pearson') |>
 mutate(doc_i=factor(doc_i,levels=rev(rownames(r))),doc_j=factor(doc_j,levels=colnames(r)))
p <- ggplot(heat,aes(doc_j,doc_i,fill=pearson))+geom_tile(color='white',linewidth=.3)+coord_fixed()+
 scale_fill_gradient2(low='#29456B',mid='#FAFAF7',high='#AE3E39',midpoint=0,limits=c(-1,1))+
 geom_text(aes(label=sprintf('%.2f',pearson)),size=2.5)+
 labs(title='Shared vocabulary across political-science articles',subtitle='Pearson correlation of count vectors; terms appearing \u22655 times across the corpus',x=NULL,y=NULL,fill='Pearson r',caption='A: alliances/bandwagoning | B: incumbent survival | C: external demands/compliance. Similarity is not causation.')+
 theme(panel.grid=element_blank(),axis.text=element_text(size=10))
ggsave(fig('pearson_heatmap.png'),p,width=12,height=11,dpi=180)
cl <- hclust(as.dist(1-r),method='average')
write_csv(tibble(order=seq_along(cl$order),doc_id=rownames(r)[cl$order]),out('exploratory_cluster_order.csv'))
p <- ggplot(counts |> filter(group!='S'),aes(doc_id,cleaned_tokens,fill=group))+geom_col()+scale_fill_manual(values=pal)+
 labs(title='Corpus length after lexical cleaning',subtitle='Length and article style are potential confounds; three purposive groups of six',x=NULL,y='Retained word tokens')+
 theme(axis.text.x=element_text(angle=45,hjust=1))
ggsave(fig('corpus_lengths.png'),p,width=11,height=5,dpi=180)
# Additional TF-IDF comparison within the three official speeches.
sp_top <- tidy_matrix(dfm_tfidf(speeches,base=10),'tfidf') |> filter(tfidf>0) |>
 group_by(doc_id) |> arrange(desc(tfidf),word,.by_group=TRUE) |> slice_head(n=15) |> ungroup()
write_csv(sp_top,out('tfidf_top15_speeches.csv'))
sp_r <- cor(t(as.matrix(speeches)))
write_csv(as.data.frame(sp_r)|>tibble::rownames_to_column('doc_id'),out('pearson_speeches.csv'))
pp <- sp_top |> group_by(doc_id) |> slice_head(n=10) |> ungroup() |>
 mutate(label=recode(doc_id,S01='Armenia',S02='Singapore',S03='Kazakhstan'),key=paste(doc_id,word,sep='___'),key=factor(key,levels=rev(unique(key))))
p <- ggplot(pp,aes(tfidf,key))+geom_col(fill=pal['S'])+facet_wrap(~label,scales='free',nrow=1)+
 scale_y_discrete(labels=function(x)sub('.*___','',x))+
 labs(title='Three governments speaking at UNGA77 (2022)',subtitle='Speech-only TF-IDF: distinctiveness relative to these three statements',x='TF-IDF',y=NULL,caption='Different countries are not labelled as examples of the three theories.')
ggsave(fig('speech_distinctiveness.png'),p,width=12,height=5,dpi=180)
# Choose a cross-group lexical bridge for syntax, deterministically and transparently.
selected <- pair_groups |> filter(!same_group,!shared_author) |> slice_head(n=1) |>
 mutate(selection_rule='Highest Pearson cross-group pair without a shared author')
write_csv(selected,out('syntax_selection.csv'))
write_lines(c(paste('Article documents:',ndoc(articles)),paste('Untrimmed features:',nfeat(articles)),paste('Trimmed features:',nfeat(trimmed)),
 paste('Spearman rank agreement trimmed/untrimmed:',cor(sens$pearson,sens$pearson_untrimmed,method='spearman'))),out('lexical_diagnostics.txt'))
capture.output(sessionInfo(),file=out('R_session_lexical.txt'))
print(read_csv(out('pearson_extreme_pairs.csv'),show_col_types=FALSE))
print(selected)


# ---- 5. Approach 3: syntactic comparison with English-EWT UDPipe ----
# Compare the selected cross-group pair using sentence and clause proxies.
out <- function(n)file.path(root,'results',n)
selection <- read_csv(out('syntax_selection.csv'),show_col_types=FALSE)
ids <- c(selection$doc_i[1],selection$doc_j[1])
text <- vapply(ids,function(id)read_file(file.path(root,'texts',paste0(id,'.txt'))),character(1))
model <- udpipe_load_model(list.files(file.path(root,'models'),pattern='udpipe$',full.names=TRUE)[1])
annotation <- list()
for(i in seq_along(ids)) {
 cache <- out(paste0('syntax_',ids[i],'.rds'))
 hash <- unname(tools::md5sum(file.path(root,'texts',paste0(ids[i],'.txt'))))
 if(file.exists(cache) && identical(readRDS(cache)$hash,hash)) a <- readRDS(cache)$data else {
  message('Parsing full body of ',ids[i])
  a <- as.data.frame(udpipe_annotate(model,x=text[i],doc_id=ids[i]))
  saveRDS(list(hash=hash,data=a),cache)
 }
 annotation[[i]] <- a
}
a <- bind_rows(annotation)
stopifnot(nrow(a)>1000,all(ids %in% a$doc_id))
write_csv(a,out('syntax_annotations.csv.gz'))
# Multiword-token placeholders are not syntactic nodes. Distinguish paragraph IDs.
s <- a |> filter(grepl('^[0-9]+$',token_id),!is.na(upos)) |>
 mutate(is_word=!upos %in% c('PUNCT','SYM'),
 is_clause=upos %in% c('VERB','AUX') & grepl('VerbForm=Fin',coalesce(feats,'')),
 is_dep=dep_rel %in% c('advcl','ccomp','xcomp','acl','acl:relcl'),
 is_coord=dep_rel %in% c('conj','cc'),
 is_nominal=dep_rel %in% c('amod','nmod','compound','appos')) |>
 group_by(doc_id,paragraph_id,sentence_id) |>
 summarise(sentence=first(sentence),words=sum(is_word),clauses=sum(is_clause),dependent=sum(is_dep),coordination=sum(is_coord),nominals=sum(is_nominal),.groups='drop')
write_csv(s,out('syntax_sentences.csv'))
metrics <- function(s) s |> group_by(doc_id) |>
 summarise(sentences=n(),words=sum(words),clause_proxies=sum(clauses),dependent_relations=sum(dependent),coordination_relations=sum(coordination),nominal_relations=sum(nominals),
 MLS=words/sentences,C_per_S=sum(clauses)/n(),DC_per_C=sum(dependent)/sum(clauses),DC_per_S=sum(dependent)/n(),
 Coord_per_C=sum(coordination)/sum(clauses),Coord_per_S=sum(coordination)/n(),CN_per_C=sum(nominals)/sum(clauses),CN_per_S=sum(nominals)/n(),.groups='drop')
m <- metrics(s)
stopifnot(all(m$MLS>5 & m$MLS<100),all(abs(m$MLS-m$words/m$sentences)<1e-10))
write_csv(m,out('syntax_summary.csv'))
# Robustness: omit short fragments (<5 words) and sentences lacking finite-verb proxies.
write_csv(metrics(filter(s,words>=5,clauses>0)),out('syntax_prose_sensitivity.csv'))
diagnostics <- s |> group_by(doc_id) |> summarise(sentences=n(),zero_finite_verb=sum(clauses==0),short_fragments=sum(words<5),over_80_words=sum(words>80),.groups='drop')
write_csv(diagnostics,out('syntax_diagnostics.csv'))
# Candidate examples retained for manual contextual inspection, not cherry-picked evidence.
examples <- s |> filter(words>=15,words<=45,clauses>=1) |> group_by(doc_id) |>
 mutate(distance_from_typical=abs(words-median(words))+abs(nominals-median(nominals))+abs(dependent-median(dependent))) |>
 arrange(distance_from_typical) |> slice_head(n=10) |> ungroup()
write_csv(examples,out('syntax_example_candidates.csv'))
long <- m |> select(doc_id,MLS,C_per_S,DC_per_C,DC_per_S,Coord_per_C,Coord_per_S,CN_per_C,CN_per_S) |>
 pivot_longer(-doc_id,names_to='measure',values_to='value')
long$measure <- factor(long$measure,levels=c('MLS','C_per_S','DC_per_C','DC_per_S','Coord_per_C','Coord_per_S','CN_per_C','CN_per_S'),
 labels=c('Words / sentence','Finite-verb proxies / sentence','Dependent relations / proxy','Dependent relations / sentence','Coordination relations / proxy','Coordination relations / sentence','Nominal modifiers / proxy','Nominal modifiers / sentence'))
p <- ggplot(long,aes(doc_id,value,fill=doc_id))+geom_col(width=.65)+geom_text(aes(label=sprintf('%.2f',value)),vjust=-.35,size=3.5)+
 facet_wrap(~measure,scales='free_y',ncol=4)+scale_fill_manual(values=c('#A04A34','#5E6F3D'))+
 scale_y_continuous(expand=expansion(mult=c(0,.16)))+
 labs(title=paste('Syntax of a cross-group lexical bridge:',paste(ids,collapse=' and ')),
 subtitle='Strongest cross-group pair without a shared author; full body texts parsed with English-EWT UDPipe',
 x=NULL,y=NULL,caption='Week05 heuristic proxies, not exact linguistic clause counts. Facets use different vertical scales.')+
 theme_minimal(base_size=11)+theme(legend.position='none',panel.grid.minor=element_blank(),strip.text=element_text(size=9),plot.title=element_text(face='bold'))
ggsave(file.path(root,'figures/syntax_comparison.png'),p,width=14,height=7,dpi=180)
capture.output(sessionInfo(),file=out('R_session_syntax.txt'))
print(m)


# ---- 6. Extension: speech dictionary and KWIC context ----
# Exploratory topic indicators, not validated policy or emotion classifications.
out <- function(x)file.path(root,'results',x)
meta <- read_csv(file.path(root,'metadata.csv'),show_col_types=FALSE) |> filter(group=='S')
txt <- setNames(vapply(meta$text_file,function(p)read_file(file.path(root,p)),character(1)),meta$doc_id)
country <- c(S01='Armenia',S02='Singapore',S03='Kazakhstan')
# Explicit seeds, chosen to operationalize three topics. No outcome-driven tuning.
lex <- list(
  Sovereignty=c('sovereignty','sovereign','independence','independent','territorial integrity','non interference','self determination'),
  Cooperation=c('cooperation','cooperate','cooperative','partnership','partnerships','multilateral','multilateralism','solidarity','dialogue'),
  Security=c('security','threat','threats','aggression','conflict','conflicts','war','invasion','violence','peace'))
write_csv(bind_rows(lapply(names(lex),function(k)tibble(category=k,term=lex[[k]]))),out('speech_dictionary.csv'))
tokenize <- function(x) tokens(x,remove_punct=TRUE,remove_symbols=TRUE,remove_numbers=TRUE,split_hyphens=TRUE) |> tokens_tolower()
tok <- tokenize(txt)
# Keep stopwords, including within multiword phrases. Denominator: all tokens retained here.
denom <- ntoken(tok)
hits <- bind_rows(lapply(names(lex),function(k){
  z <- as.data.frame(kwic(tok,pattern=phrase(lex[[k]]),valuetype='fixed',window=12))
  z$category <- rep(k,nrow(z));z
}))
write_excel_csv(hits,out('speech_kwic_LOCAL_REVIEW.csv'))
counts <- hits |> count(docname,category,name='matches') |>
 complete(docname=names(txt),category=names(lex),fill=list(matches=0)) |>
 mutate(total_tokens=as.numeric(denom[docname]),per_1000=1000*matches/total_tokens,country=unname(country[docname]))
stopifnot(all(is.finite(counts$per_1000)),nrow(counts)==9)
write_csv(counts,out('speech_dictionary_rates.csv'))
p <- ggplot(counts,aes(x=category,y=per_1000,fill=country))+geom_col(position=position_dodge(.8),width=.7)+
 geom_text(aes(label=sprintf('%.1f',per_1000)),position=position_dodge(.8),vjust=-.4,size=3.5)+
 scale_fill_manual(values=c(Armenia='#986044',Kazakhstan='#5E6F3D',Singapore='#236B8E'))+
 labs(title='Sovereignty, cooperation and security vocabulary',subtitle='Three UNGA77 speeches | exploratory custom dictionary, not policy classification',
 x=NULL,y='Dictionary matches per 1,000 retained tokens',fill=NULL,
 caption='Exact terms and phrases; stopwords retained in denominator. Security includes peace; categories are not sentiment scores.')+
 theme_minimal(base_size=12)+theme(legend.position='bottom',panel.grid.minor=element_blank())+expand_limits(y=max(counts$per_1000)*1.12)
ggsave(file.path(root,'figures/speech_dictionary_rates.png'),p,width=11,height=6,dpi=180)
# Human-review worksheet: automatic candidate flags only, no claim of manual coding.
sentences <- bind_rows(lapply(names(txt),function(id){
 ss <- as.character(tokens(txt[[id]],what='sentence')[[1]])
 tibble(doc_id=id,sentence_id=seq_along(ss),sentence=ss)
})) |> mutate(unit_id=paste(doc_id,sentence_id,sep='_'))
st <- tokenize(setNames(sentences$sentence,sentences$unit_id))
flags <- bind_rows(lapply(names(lex),function(k){
 z <- as.data.frame(kwic(st,pattern=phrase(lex[[k]]),valuetype='fixed',window=0))
 tibble(unit_id=as.character(z$docname),category=k)
})) |> distinct() |> group_by(unit_id) |> summarise(candidate_categories=paste(sort(category),collapse='; '),.groups='drop')
review <- sentences |> group_by(doc_id) |> mutate(previous_sentence=lag(sentence,default=''),next_sentence=lead(sentence,default='')) |> ungroup() |>
 left_join(flags,by='unit_id') |> mutate(candidate_categories=coalesce(candidate_categories,''),
 status=if_else(candidate_categories=='','unmatched; inspect for missed language','dictionary candidate; not human coded'),
 human_labels='',actor_target='',evidence_and_reason='',uncertain='',reviewer='')
write_excel_csv(review,out('speech_sentence_review_LOCAL.csv'))
coverage <- review |> group_by(doc_id) |> summarise(sentences=n(),candidate_sentences=sum(candidate_categories!=''),.groups='drop')
write_csv(coverage,out('speech_dictionary_coverage.csv'))
capture.output(sessionInfo(),file=out('R_session_speech_exploration.txt'))
print(counts);print(coverage)


# ---- 7. Extension: summarize the 30 student-coded text units ----
# Read the completed coding file; do not resample or change student labels.
r <- file.path(root,'results')
input <- file.path(r,'speech_review_30_LOCAL.csv')
if (!file.exists(input)) input <- file.path(r,'speech_review_30_labels_only.csv')
x <- read_csv(input,col_types=cols(.default=col_character()))
labels <- c('\u4e3b\u5f20\u4e3b\u6743\uff0f\u81ea\u4e3b','\u5bfb\u6c42\uff0f\u627f\u8bfa\u5408\u4f5c','\u8868\u8fbe\u5a01\u80c1\uff0f\u8c34\u8d23','\u5176\u4ed6','\u4e0d\u786e\u5b9a')
stopifnot(nrow(x)==30,all(x$human_labels %in% as.character(1:5)),all(x$uncertain %in% c('1','2')))
x <- x |> mutate(label=labels[as.integer(human_labels)],confidence=if_else(uncertain=='1','\u62ff\u4e0d\u51c6','\u62ff\u5f97\u51c6'))
write_excel_csv(x,file.path(r,'speech_review_30_decoded_LOCAL.csv'))
summary <- x |> count(label,confidence,name='sentences') |> complete(label=labels,confidence=c('\u62ff\u4e0d\u51c6','\u62ff\u5f97\u51c6'),fill=list(sentences=0))
write_excel_csv(summary,file.path(r,'speech_review_30_summary.csv'))
write_excel_csv(x |> count(sample_stratum,label,confidence,name='sentences'),file.path(r,'speech_review_30_by_stratum.csv'))
print(summary)
english <- c('Sovereignty / autonomy','Cooperation','Threat / condemnation','Other','Undetermined')
x <- x |> mutate(category=factor(english[as.integer(human_labels)],levels=english),
 country=factor(doc_id,levels=c('S01','S02','S03'),labels=c('Armenia','Singapore','Kazakhstan')),
 certainty=factor(uncertain,levels=c('2','1'),labels=c('Coder confident','Coder uncertain')))
theme_set(theme_minimal(base_size=12)+theme(panel.grid.minor=element_blank(),legend.position='bottom'))
plot_counts <- function(data,facet,filename,title) {
 z <- data |> count(.data[[facet]],category,certainty,name='sentences') |>
  complete(!!rlang::sym(facet),category,certainty,fill=list(sentences=0))
 p <- ggplot(z,aes(x=category,y=sentences,fill=certainty))+geom_col(width=.7)+
 facet_wrap(stats::as.formula(paste('~',facet)))+coord_flip()+
 scale_fill_manual(values=c('Coder confident'='#236B8E','Coder uncertain'='#E3AD54'))+
 scale_y_continuous(breaks=0:10,limits=c(0,10))+labs(title=title,
 subtitle='30 stratified review units | original student labels retained without AI relabelling',
 x=NULL,y='Number of sampled units',fill=NULL,
 caption='Each country: 5 dictionary-hit + 5 non-hit units. Counts describe this sample, not full-speech proportions.')
 ggsave(file.path(root,'figures',filename),p,width=12,height=6,dpi=180)
 z
}
country_counts <- plot_counts(x,'country','speech_human_review_country.png','How the student classified the sampled speech units')
strata <- x |> mutate(retrieval=if_else(sample_stratum=='matched','Dictionary hit (15 units)','No dictionary hit (15 units)'))
stratum_counts <- plot_counts(strata,'retrieval','speech_human_review_retrieval.png','Dictionary retrieval and contextual human interpretation')
write_csv(country_counts,file.path(r,'speech_human_review_country.csv'))
write_csv(stratum_counts,file.path(r,'speech_human_review_retrieval.csv'))
print(x |> count(sample_stratum,category))
# These are analytical units from automatic sentence splitting, not guaranteed single sentences.
capture.output(sessionInfo(),file=file.path(r,'R_session_human_review.txt'))
message("ALL ANALYSES COMPLETE. Results: ",file.path(root,"results")," | Figures: ",file.path(root,"figures"))
