# This file was taken from:
# https://github.com/elikip/bist-parser/blob/master/bmstparser/src/utils/eval.pl
# License: https://github.com/elikip/bist-parser/blob/master/LICENSE

#!/usr/bin/env perl

# Author: Yuval Krymolowski
# Addition of precision and recall 
#   and of frame confusion list: Sabine Buchholz
# Addition of DEPREL + ATTACHMENT:
#   Prokopis Prokopidis (prokopis at ilsp dot gr)
# Acknowledgements: 
#   to Markus Kuhn for suggesting the use of 
#   the Unicode category property

if ($] < 5.008001)
{
  printf STDERR <<EOM

 This script requires PERL 5.8.1 for running.
 The new version is needed for proper handling
 of Unicode characters.

 Please obtain a new version or contact the shared task team
 if you are unable to upgrade PERL.

EOM
;
  exit(1) ;
}

require Encode;

use strict ;
use warnings;
use Getopt::Std ;

my ($usage) = <<EOT

  CoNLL-X evaluation script:

   [perl] eval.pl [OPTIONS] -g <gold standard> -s <system output>

  This script evaluates a system output with respect to a gold standard.
  Both files should be in UTF-8 encoded CoNLL-X tabular format.

  Punctuation tokens (those where all characters have the Unicode
  category property "Punctuation") are ignored for scoring (unless the
  -p flag is used).

  The output breaks down the errors according to their type and context.

  Optional parameters:
     -o FILE : output: print output to FILE (default is standard output)
     -q : quiet:       only print overall performance, without the details
     -b : evalb:       produce output in a format similar to evalb 
                       (http://nlp.cs.nyu.edu/evalb/); use together with -q
     -p : punctuation: also score on punctuation (default is not to score on it)
     -v : version:     show the version number
     -h : help:        print this help text and exit

EOT
;

my ($line_num) ;
my ($sep) = '0x01' ;

my ($START) = '.S' ;
my ($END) = '.E' ;

my ($con_err_num) = 3 ;
my ($freq_err_num) = 10 ;
my ($spec_err_loc_con) = 8 ;

################################################################################
###                              subfunctions                                ###
################################################################################

# Whether a string consists entirely of characters with the Unicode
# category property "Punctuation" (see "man perlunicode")
sub is_uni_punct
{
  my ($word) = @_ ;

  return scalar(Encode::decode_utf8($word)=~ /^\p{Punctuation}+$/) ;
}

# The length of a unicode string, excluding non-spacing marks
# (for example vowel marks in Arabic)

sub uni_len
{
  my ($word) = @_ ;
  my ($ch, $l) ;

  $l = 0 ;
  foreach $ch (split(//,  Encode::decode_utf8($word)))
  {
    if ($ch !~ /^\p{NonspacingMark}/)
    {
      $l++ ;
    }
  }

  return $l ;
}

sub filter_context_counts
{ # filter_context_counts

  my ($vec, $num, $max_len) = @_ ;
  my ($con, $l, $thresh) ;

  $thresh = (sort {$b <=> $a} values %{$vec})[$num-1] ;

  foreach $con (keys %{$vec})
  {
    if (${$vec}{$con} < $thresh)
    {
      delete ${$vec}{$con} ;
      next ;
    }

    $l = uni_len($con) ;

    if ($l > ${$max_len})
    {
      ${$max_len} = $l ;
    }
  }

} # filter_context_counts

sub print_context
{ # print_context

  my ($counts, $counts_pos, $max_con_len, $max_con_pos_len) = @_ ;
  my (@v_con, @v_con_pos, $con, $con_pos, $i, $n) ;

  printf OUT "  %-*s | %-4s | %-4s | %-4s | %-4s", $max_con_pos_len, 'CPOS', 'any', 'head', 'dep', 'both' ;
  printf OUT "  ||" ;
  printf OUT "  %-*s | %-4s | %-4s | %-4s | %-4s", $max_con_len, 'word', 'any', 'head', 'dep', 'both' ;
  printf OUT "\n" ;
  printf OUT "  %s-+------+------+------+-----", '-' x $max_con_pos_len;
  printf OUT "--++" ;
  printf OUT "--%s-+------+------+------+-----", '-' x $max_con_len;
  printf OUT "\n" ;

  @v_con = sort {${$counts}{tot}{$b} <=> ${$counts}{tot}{$a}} keys %{${$counts}{tot}} ;
  @v_con_pos = sort {${$counts_pos}{tot}{$b} <=> ${$counts_pos}{tot}{$a}} keys %{${$counts_pos}{tot}} ;

  $n = scalar @v_con ;
  if (scalar @v_con_pos > $n)
  {
    $n = scalar @v_con_pos ;
  }

  foreach $i (0 .. $n-1)
  {
    if (defined $v_con_pos[$i])
    {
      $con_pos = $v_con_pos[$i] ;
      printf OUT "  %-*s | %4d | %4d | %4d | %4d",
	$max_con_pos_len, $con_pos, ${$counts_pos}{tot}{$con_pos},
	  ${$counts_pos}{err_head}{$con_pos}, ${$counts_pos}{err_dep}{$con_pos},
	    ${$counts_pos}{err_dep}{$con_pos}+${$counts_pos}{err_head}{$con_pos}-${$counts_pos}{tot}{$con_pos} ;
    }
    else
    {
      printf OUT "  %-*s | %4s | %4s | %4s | %4s",
	$max_con_pos_len, ' ', ' ', ' ', ' ', ' ' ;
    }

    printf OUT "  ||" ;

    if (defined $v_con[$i])
    {
      $con = $v_con[$i] ;
      printf OUT "  %-*s | %4d | %4d | %4d | %4d",
	$max_con_len+length($con)-uni_len($con), $con, ${$counts}{tot}{$con},
	  ${$counts}{err_head}{$con}, ${$counts}{err_dep}{$con},
	    ${$counts}{err_dep}{$con}+${$counts}{err_head}{$con}-${$counts}{tot}{$con} ;
    }
    else
    {
      printf OUT "  %-*s | %4s | %4s | %4s | %4s",
	$max_con_len, ' ', ' ', ' ', ' ', ' ' ;
    }

    printf OUT "\n" ;
  }

  printf OUT "  %s-+------+------+------+-----", '-' x $max_con_pos_len;
  printf OUT "--++" ;
  printf OUT "--%s-+------+------+------+-----", '-' x $max_con_len;
  printf OUT "\n" ;

  printf OUT "\n\n" ;

} # print_context

sub num_as_word
{
  my ($num) = @_ ;

  $num = abs($num) ;

  if ($num == 1)
  {
    return ('one word') ;
  }
  elsif ($num == 2)
  {
    return ('two words') ;
  }
  elsif ($num == 3)
  {
    return ('three words') ;
  }
  elsif ($num == 4)
  {
    return ('four words') ;
  }
  else
  {
    return ($num.' words') ;
  }
}

sub describe_err
{ # describe_err

  my ($head_err, $head_aft_bef, $dep_err) = @_ ;
  my ($dep_g, $dep_s, $desc) ;
  my ($head_aft_bef_g, $head_aft_bef_s) = split(//, $head_aft_bef) ;

  if ($head_err eq '-')
  {
    $desc = 'correct head' ;

    if ($head_aft_bef_s eq '0')
    {
      $desc .= ' (0)' ;
    }
    elsif ($head_aft_bef_s eq 'e')
    {
      $desc .= ' (the focus word)' ;
    }
    elsif ($head_aft_bef_s eq 'a')
    {
      $desc .= ' (after the focus word)' ;
    }
    elsif ($head_aft_bef_s eq 'b')
    {
      $desc .= ' (before the focus word)' ;
    }
  }
  elsif ($head_aft_bef_s eq '0')
  {
    $desc = 'head = 0 instead of ' ;
    if ($head_aft_bef_g eq 'a')
    {
      $desc.= 'after ' ;
    }
    if ($head_aft_bef_g eq 'b')
    {
      $desc.= 'before ' ;
    }
    $desc .= 'the focus word' ;
  }
  elsif ($head_aft_bef_g eq '0')
  {
    $desc = 'head is ' ;
    if ($head_aft_bef_g eq 'a')
    {
      $desc.= 'after ' ;
    }
    if ($head_aft_bef_g eq 'b')
    {
      $desc.= 'before ' ;
    }
    $desc .= 'the focus word instead of 0' ;
  }
  else
  {
    $desc = num_as_word($head_err) ;
    if ($head_err < 0)
    {
      $desc .= ' before' ;
    }
    else
    {
      $desc .= ' after' ;
    }

    $desc = 'head '.$desc.' the correct head ' ;

    if ($head_aft_bef_s eq '0')
    {
      $desc .= '(0' ;
    }
    elsif ($head_aft_bef_s eq 'e')
    {
      $desc .= '(the focus word' ;
    }
    elsif ($head_aft_bef_s eq 'a')
    {
      $desc .= '(after the focus word' ;
    }
    elsif ($head_aft_bef_s eq 'b')
    {
      $desc .= '(before the focus word' ;
    }

    if ($head_aft_bef_g ne $head_aft_bef_s)
    {
      $desc .= ' instead of' ;
      if ($head_aft_bef_s eq '0')
      {
	$desc .= '0' ;
      }
      elsif ($head_aft_bef_s eq 'e')
      {
	$desc .= 'the focus word' ;
      }
      elsif ($head_aft_bef_s eq 'a')
      {
	$desc .= 'after the focus word' ;
      }
      elsif ($head_aft_bef_s eq 'b')
      {
	$desc .= 'before the focus word' ;
      }
    }

    $desc .= ')' ;
  }

  $desc .= ', ' ;

  if ($dep_err eq '-')
  {
    $desc .= 'correct dependency' ;
  }
  else
  {
    ($dep_g, $dep_s) = ($dep_err =~ /^(.*)->(.*)$/) ;
    $desc .= sprintf('dependency "%s" instead of "%s"', $dep_s, $dep_g) ;
  }

  return($desc) ;

} # describe_err

sub get_context
{ # get_context

  my ($sent, $i_w) = @_ ;
  my ($w_2, $w_1, $w1, $w2) ;
  my ($p_2, $p_1, $p1, $p2) ;

  if ($i_w >= 2)
  {
    $w_2 = ${${$sent}[$i_w-2]}{word} ;
    $p_2 = ${${$sent}[$i_w-2]}{pos} ;
  }
  else
  {
    $w_2 = $START ;
    $p_2 = $START ;
  }

  if ($i_w >= 1)
  {
    $w_1 = ${${$sent}[$i_w-1]}{word} ;
    $p_1 = ${${$sent}[$i_w-1]}{pos} ;
  }
  else
  {
    $w_1 = $START ;
    $p_1 = $START ;
  }

  if ($i_w <= scalar @{$sent}-2)
  {
    $w1 = ${${$sent}[$i_w+1]}{word} ;
    $p1 = ${${$sent}[$i_w+1]}{pos} ;
  }
  else
  {
    $w1 = $END ;
    $p1 = $END ;
  }

  if ($i_w <= scalar @{$sent}-3)
  {
    $w2 = ${${$sent}[$i_w+2]}{word} ;
    $p2 = ${${$sent}[$i_w+2]}{pos} ;
  }
  else
  {
    $w2 = $END ;
    $p2 = $END ;
  }

  return ($w_2, $w_1, $w1, $w2, $p_2, $p_1, $p1, $p2) ;

} # get_context

sub read_sent
{ # read_sent

  my ($sent_gold, $sent_sys) = @_ ;
  my ($line_g, $line_s, $new_sent) ;
  my (%fields_g, %fields_s) ;

  $new_sent = 1 ;

  @{$sent_gold} = () ;
  @{$sent_sys} = () ;

  while (1)
  { # main reading loop

    $line_g = <GOLD> ;
    $line_s = <SYS> ;

    $line_num++ ;

    # system output has fewer lines than gold standard
    if ((defined $line_g) && (! defined $line_s))
    {
	printf STDERR "line mismatch, line %d:\n", $line_num ;
	printf STDERR " gold: %s", $line_g ;
	printf STDERR " sys : past end of file\n" ;
	exit(1) ;
    }

    # system output has more lines than gold standard
    if ((! defined $line_g) && (defined $line_s))
    {
	printf STDERR "line mismatch, line %d:\n", $line_num ;
	printf STDERR " gold: past end of file\n" ;
	printf STDERR " sys : %s", $line_s ;
	exit(1) ;
    }
    
    # end of file reached for both
    if ((! defined $line_g) && (! defined $line_s))
    {
	return (1) ;
    }

    # one contains end of sentence but other one does not
    if (($line_g =~ /^\s+$/) != ($line_s =~ /^\s+$/))
    {
      printf STDERR "line mismatch, line %d:\n", $line_num ;
      printf STDERR " gold: %s", $line_g ;
      printf STDERR " sys : %s", $line_s ;
      exit(1) ;
    }

    # end of sentence reached
    if ($line_g =~ /^\s+$/)
    {
	return(0) ;
    }

    # now both lines contain information

    if ($new_sent)
    {
      $new_sent = 0 ;
    }

    # 'official' column names
    # options.output = ['id','form','lemma','cpostag','postag',
    #                   'feats','head','deprel','phead','pdeprel']

    @fields_g{'word', 'pos', 'head', 'dep'} = (split (/\s+/, $line_g))[1, 3, 6, 7] ;

    push @{$sent_gold}, { %fields_g } ;

    @fields_s{'word', 'pos', 'head', 'dep'} = (split (/\s+/, $line_s))[1, 3, 6, 7] ;

    if (($fields_g{word} ne $fields_s{word})
	||
	($fields_g{pos} ne $fields_s{pos}))
    {
      printf STDERR "Word/pos mismatch, line %d:\n", $line_num ;
      printf STDERR " gold: %s", $line_g ;
      printf STDERR " sys : %s", $line_s ;
      #exit(1) ;
    }

    push @{$sent_sys}, { %fields_s } ;

  } # main reading loop
  
} # read_sent

################################################################################
###                                  main                                    ###
################################################################################

our ($opt_g, $opt_s, $opt_o, $opt_h, $opt_v, $opt_q, $opt_p, $opt_b) ;

my ($sent_num, $eof, $word_num, @err_sent) ;
my (@sent_gold, @sent_sys, @starts) ;
my ($word, $pos, $wp, $head_g, $dep_g, $head_s, $dep_s) ;
my (%counts, $err_head, $err_dep, $con, $con1, $con_pos, $con_pos1, $thresh) ;
my ($head_err, $dep_err, @cur_err, %err_counts, $err_counter, $err_desc) ;
my ($loc_con, %loc_con_err_counts, %err_desc) ;
my ($head_aft_bef_g, $head_aft_bef_s, $head_aft_bef) ;
my ($con_bef, $con_aft, $con_bef_2, $con_aft_2, @bits, @e_bits, @v_con, @v_con_pos) ;
my ($con_pos_bef, $con_pos_aft, $con_pos_bef_2, $con_pos_aft_2) ;
my ($max_word_len, $max_pos_len, $max_con_len, $max_con_pos_len) ;
my ($max_word_spec_len, $max_con_bef_len, $max_con_aft_len) ;
my (%freq_err, $err) ;

my ($i, $j, $i_w, $l, $n_args) ;
my ($w_2, $w_1, $w1, $w2) ;
my ($wp_2, $wp_1, $wp1, $wp2) ;
my ($p_2, $p_1, $p1, $p2) ;

my ($short_output) ;
my ($score_on_punct) ;
$counts{punct} = 0; # initialize

getopts("g:o:s:qvhpb") ;

if (defined $opt_v)
{
    my $id = '$Id: eval.pl,v 1.9 2006/05/09 20:30:01 yuval Exp $';
    my @parts = split ' ',$id;
    print "Version $parts[2]\n";
    exit(0);
}

if ((defined $opt_h) || ((! defined $opt_g) && (! defined $opt_s)))
{
  die $usage ;
}

if (! defined $opt_g)
{
  die "Gold standard file (-g) missing\n" ;
}

if (! defined $opt_s)
{
  die "System output file (-s) missing\n" ;
}

if (! defined $opt_o)
{
  $opt_o = '-' ;
}

if (defined $opt_q)
{
    $short_output = 1 ;
} else {
    $short_output = 0 ;
}

if (defined $opt_p)
{
    $score_on_punct = 1 ;
} else {
    $score_on_punct = 0 ;
}

$line_num = 0 ;
$sent_num = 0 ;
$eof = 0 ;

@err_sent = () ;
@starts = () ;

%{$err_sent[0]} = () ;

$max_pos_len = length('CPOS') ;

################################################################################
###                              reading input                               ###
################################################################################

open (GOLD, "<$opt_g") || die "Could not open gold standard file $opt_g\n" ;
open (SYS,  "<$opt_s") || die "Could not open system output file $opt_s\n" ;
open (OUT,  ">$opt_o") || die "Could not open output file $opt_o\n" ;


if (defined $opt_b) {  # produce output similar to evalb
    print OUT "     Sent.          Attachment      Correct        Scoring          \n";
    print OUT "    ID Tokens  -   Unlab. Lab.   HEAD HEAD+DEPREL   tokens   - - - -\n";
    print OUT "  ============================================================================\n";
}


while (! $eof)
{ # main reading loop

  $starts[$sent_num] = $line_num+1 ;
  $eof = read_sent(\@sent_gold, \@sent_sys) ;

  $sent_num++ ;

  %{$err_sent[$sent_num]} = () ;
  $word_num = scalar @sent_gold ;

  # for accuracy per sentence
  my %sent_counts = ( tot      => 0,
		      err_any  => 0,
		      err_head => 0
		      ); 

  # printf "$sent_num $word_num\n" ;

  my @frames_g = ('** '); # the initial frame for the virtual root
  my @frames_s = ('** '); # the initial frame for the virtual root
  foreach $i_w (0 .. $word_num-1)
  { # loop on words
      push @frames_g, ''; # initialize
      push @frames_s, ''; # initialize
  }

  foreach $i_w (0 .. $word_num-1)
  { # loop on words

    ($word, $pos, $head_g, $dep_g)
      = @{$sent_gold[$i_w]}{'word', 'pos', 'head', 'dep'} ;
    $wp = $word.' / '.$pos ;

    # printf "%d: %s %s %s %s\n", $i_w,  $word, $pos, $head_g, $dep_g ;

    if ((! $score_on_punct) && is_uni_punct($word))
    {
      $counts{punct}++ ;
      # ignore punctuations
      next ;
    }

    if (length($pos) > $max_pos_len)
    {
      $max_pos_len = length($pos) ;
    }

    ($head_s, $dep_s) = @{$sent_sys[$i_w]}{'head', 'dep'} ;

    $counts{tot}++ ;
    $counts{word}{$wp}{tot}++ ;
    $counts{pos}{$pos}{tot}++ ;
    $counts{head}{$head_g-$i_w-1}{tot}++ ;

    # for frame confusions
    # add child to frame of parent
    $frames_g[$head_g] .= "$dep_g ";
    $frames_s[$head_s] .= "$dep_s ";
    # add to frame of token itself
    $frames_g[$i_w+1] .= "*$dep_g* "; # $i_w+1 because $i_w starts counting at zero
    $frames_s[$i_w+1] .= "*$dep_g* ";

    # for precision and recall of DEPREL
    $counts{dep}{$dep_g}{tot}++ ;     # counts for gold standard deprels
    $counts{dep2}{$dep_g}{$dep_s}++ ; # counts for confusions
    $counts{dep_s}{$dep_s}{tot}++ ;   # counts for system deprels
    $counts{all_dep}{$dep_g} = 1 ;    # list of all deprels that occur ...
    $counts{all_dep}{$dep_s} = 1 ;    # ... in either gold or system output

    # for precision and recall of HEAD direction
    my $dir_g;
    if ($head_g == 0) {
	$dir_g = 'to_root';
    } elsif ($head_g < $i_w+1) { # $i_w+1 because $i_w starts counting at zero
                                 # also below
	$dir_g = 'left';
    } elsif ($head_g > $i_w+1) {
	$dir_g = 'right';
    } else {
        # token links to itself; should never happen in correct gold standard
	$dir_g = 'self'; 
    }
    my $dir_s;
    if ($head_s == 0) {
	$dir_s = 'to_root';
    } elsif ($head_s < $i_w+1) {
	$dir_s = 'left';
    } elsif ($head_s > $i_w+1) {
	$dir_s = 'right';
    } else {
        # token links to itself; should not happen in good system 
        # (but not forbidden in shared task)
	$dir_s = 'self'; 
    }
    $counts{dir_g}{$dir_g}{tot}++ ;   # counts for gold standard head direction
    $counts{dir2}{$dir_g}{$dir_s}++ ; # counts for confusions
    $counts{dir_s}{$dir_s}{tot}++ ;   # counts for system head direction

    # for precision and recall of HEAD distance
    my $dist_g;
    if ($head_g == 0) {
	$dist_g = 'to_root';
    } elsif ( abs($head_g - ($i_w+1)) <= 1 ) {
	$dist_g = '1'; # includes the 'self' cases
    } elsif ( abs($head_g - ($i_w+1)) <= 2 ) {
	$dist_g = '2';
    } elsif ( abs($head_g - ($i_w+1)) <= 6 ) {
	$dist_g = '3-6';
    } else {
	$dist_g = '7-...';
    }
    my $dist_s;
    if ($head_s == 0) {
	$dist_s = 'to_root';
    } elsif ( abs($head_s - ($i_w+1)) <= 1 ) {
	$dist_s = '1'; # includes the 'self' cases
    } elsif ( abs($head_s - ($i_w+1)) <= 2 ) {
	$dist_s = '2';
    } elsif ( abs($head_s - ($i_w+1)) <= 6 ) {
	$dist_s = '3-6';
    } else {
	$dist_s = '7-...';
    }
    $counts{dist_g}{$dist_g}{tot}++ ;    # counts for gold standard head distance
    $counts{dist2}{$dist_g}{$dist_s}++ ; # counts for confusions
    $counts{dist_s}{$dist_s}{tot}++ ;    # counts for system head distance


    $err_head = ($head_g ne $head_s) ; # error in head
    $err_dep = ($dep_g ne $dep_s) ;    # error in deprel

    $head_err = '-' ;
    $dep_err = '-' ;

    # for accuracy per sentence
    $sent_counts{tot}++ ;
    if ($err_dep || $err_head) {
	$sent_counts{err_any}++ ;
    }
    if ($err_head) {
	$sent_counts{err_head}++ ;
    }

    # total counts and counts for CPOS involved in errors

    if ($head_g eq '0')
    {
      $head_aft_bef_g = '0' ;
    }
    elsif ($head_g eq $i_w+1)
    {
      $head_aft_bef_g = 'e' ;
    }
    else
    {
      $head_aft_bef_g = ($head_g <= $i_w+1 ? 'b' : 'a') ;
    }

    if ($head_s eq '0')
    {
      $head_aft_bef_s = '0' ;
    }
    elsif ($head_s eq $i_w+1)
    {
      $head_aft_bef_s = 'e' ;
    }
    else
    {
      $head_aft_bef_s = ($head_s <= $i_w+1 ? 'b' : 'a') ;
    }

    $head_aft_bef = $head_aft_bef_g.$head_aft_bef_s ;

    if ($err_head)
    {
      if ($head_aft_bef_s eq '0')
      {
	$head_err = 0 ;
      }
      else
      {
	$head_err = $head_s-$head_g ;
      }

      $err_sent[$sent_num]{head}++ ;
      $counts{err_head}{tot}++ ;
      $counts{err_head}{$head_err}++ ;

      $counts{word}{err_head}{$wp}++ ;
      $counts{pos}{$pos}{err_head}{tot}++ ;
      $counts{pos}{$pos}{err_head}{$head_err}++ ;
    }

    if ($err_dep)
    {
      $dep_err = $dep_g.'->'.$dep_s ;
      $err_sent[$sent_num]{dep}++ ;
      $counts{err_dep}{tot}++ ;
      $counts{err_dep}{$dep_err}++ ;

      $counts{word}{err_dep}{$wp}++ ;
      $counts{pos}{$pos}{err_dep}{tot}++ ;
      $counts{pos}{$pos}{err_dep}{$dep_err}++ ;

      if ($err_head)
      {
	$counts{err_both}++ ;
	$counts{pos}{$pos}{err_both}++ ;
      }
    }

    ### DEPREL + ATTACHMENT
    if ((!$err_dep) && ($err_head)) {
	$counts{err_head_corr_dep}{tot}++ ;
	$counts{err_head_corr_dep}{$dep_s}++ ;
    }
    ### DEPREL + ATTACHMENT

    # counts for words involved in errors

    if (! ($err_head || $err_dep))
    {
      next ;
    }

    $err_sent[$sent_num]{word}++ ;
    $counts{err_any}++ ;
    $counts{word}{err_any}{$wp}++ ;
    $counts{pos}{$pos}{err_any}++ ;

    ($w_2, $w_1, $w1, $w2, $p_2, $p_1, $p1, $p2) = get_context(\@sent_gold, $i_w) ;

    if ($w_2 ne $START)
    {
      $wp_2 = $w_2.' / '.$p_2 ;
    }
    else
    {
      $wp_2 = $w_2 ;
    }

    if ($w_1 ne $START)
    {
      $wp_1 = $w_1.' / '.$p_1 ;
    }
    else
    {
      $wp_1 = $w_1 ;
    }

    if ($w1 ne $END)
    {
      $wp1 = $w1.' / '.$p1 ;
    }
    else
    {
      $wp1 = $w1 ;
    }

    if ($w2 ne $END)
    {
      $wp2 = $w2.' / '.$p2 ;
    }
    else
    {
      $wp2 = $w2 ;
    }

    $con_bef = $wp_1 ;
    $con_bef_2 = $wp_2.' + '.$wp_1 ;
    $con_aft = $wp1 ;
    $con_aft_2 = $wp1.' + '.$wp2 ;

    $con_pos_bef = $p_1 ;
    $con_pos_bef_2 = $p_2.'+'.$p_1 ;
    $con_pos_aft = $p1 ;
    $con_pos_aft_2 = $p1.'+'.$p2 ;

    if ($w_1 ne $START)
    {
      # do not count '.S' as a word context
      $counts{con_bef_2}{tot}{$con_bef_2}++ ;
      $counts{con_bef_2}{err_head}{$con_bef_2} += $err_head ;
      $counts{con_bef_2}{err_dep}{$con_bef_2} += $err_dep ;
      $counts{con_bef}{tot}{$con_bef}++ ;
      $counts{con_bef}{err_head}{$con_bef} += $err_head ;
      $counts{con_bef}{err_dep}{$con_bef} += $err_dep ;
    }

    if ($w1 ne $END)
    {
      # do not count '.E' as a word context
      $counts{con_aft_2}{tot}{$con_aft_2}++ ;
      $counts{con_aft_2}{err_head}{$con_aft_2} += $err_head ;
      $counts{con_aft_2}{err_dep}{$con_aft_2} += $err_dep ;
      $counts{con_aft}{tot}{$con_aft}++ ;
      $counts{con_aft}{err_head}{$con_aft} += $err_head ;
      $counts{con_aft}{err_dep}{$con_aft} += $err_dep ;
    }

    $counts{con_pos_bef_2}{tot}{$con_pos_bef_2}++ ;
    $counts{con_pos_bef_2}{err_head}{$con_pos_bef_2} += $err_head ;
    $counts{con_pos_bef_2}{err_dep}{$con_pos_bef_2} += $err_dep ;
    $counts{con_pos_bef}{tot}{$con_pos_bef}++ ;
    $counts{con_pos_bef}{err_head}{$con_pos_bef} += $err_head ;
    $counts{con_pos_bef}{err_dep}{$con_pos_bef} += $err_dep ;

    $counts{con_pos_aft_2}{tot}{$con_pos_aft_2}++ ;
    $counts{con_pos_aft_2}{err_head}{$con_pos_aft_2} += $err_head ;
    $counts{con_pos_aft_2}{err_dep}{$con_pos_aft_2} += $err_dep ;
    $counts{con_pos_aft}{tot}{$con_pos_aft}++ ;
    $counts{con_pos_aft}{err_head}{$con_pos_aft} += $err_head ;
    $counts{con_pos_aft}{err_dep}{$con_pos_aft} += $err_dep ;

    $err = $head_err.$sep.$head_aft_bef.$sep.$dep_err ;
    $freq_err{$err}++ ;

  } # loop on words

  foreach $i_w (0 .. $word_num) # including one for the virtual root
  { # loop on words
      if ($frames_g[$i_w] ne $frames_s[$i_w]) {
	  $counts{frame2}{"$frames_g[$i_w]/ $frames_s[$i_w]"}++ ;
      }
  }

  if (defined $opt_b) { # produce output similar to evalb
      if ($word_num > 0) {
	  my ($unlabeled,$labeled) = ('NaN', 'NaN');
	  if ($sent_counts{tot} > 0) { # there are scoring tokens
	      $unlabeled = 100-$sent_counts{err_head}*100.0/$sent_counts{tot};
	      $labeled   = 100-$sent_counts{err_any} *100.0/$sent_counts{tot};
	  }
	  printf OUT "  %4d %4d    0  %6.2f %6.2f  %4d    %4d        %4d    0 0 0 0\n", 
	  $sent_num, $word_num, 
	  $unlabeled, $labeled, 
	  $sent_counts{tot}-$sent_counts{err_head}, 
	  $sent_counts{tot}-$sent_counts{err_any}, 
	  $sent_counts{tot},;
      }
  }

} # main reading loop

################################################################################
###                             printing output                              ###
################################################################################

if (defined $opt_b) {  # produce output similar to evalb
    print OUT "\n\n";
}
printf OUT "  Labeled   attachment score: %d / %d * 100 = %.2f %%\n", 
    $counts{tot}-$counts{err_any},      $counts{tot}, 100-$counts{err_any}*100.0/$counts{tot} ;
printf OUT "  Unlabeled attachment score: %d / %d * 100 = %.2f %%\n", 
    $counts{tot}-$counts{err_head}{tot}, $counts{tot}, 100-$counts{err_head}{tot}*100.0/$counts{tot} ;
printf OUT "  Label accuracy score:       %d / %d * 100 = %.2f %%\n", 
    $counts{tot}-$counts{err_dep}{tot}, $counts{tot}, 100-$counts{err_dep}{tot}*100.0/$counts{tot} ;

if ($short_output)
{
    exit(0) ;
}
printf OUT "\n  %s\n\n", '=' x 80 ;
printf OUT "  Evaluation of the results in %s\n  vs. gold standard %s:\n\n", $opt_s, $opt_g ;

printf OUT "  Legend: '%s' - the beginning of a sentence, '%s' - the end of a sentence\n\n", $START, $END ;

printf OUT "  Number of non-scoring tokens: $counts{punct}\n\n";

printf OUT "  The overall accuracy and its distribution over CPOSTAGs\n\n" ;
printf OUT "%s\n", "  -----------+-------+-------+------+-------+------+-------+-------" ;

printf OUT "  %-10s | %-5s | %-5s |   %%  | %-5s |   %%  | %-5s |   %%\n",
  'Accuracy', 'words', 'right', 'right', 'both' ;
printf OUT "  %-10s | %-5s | %-5s |      | %-5s |      | %-5s |\n",
  ' ', ' ', 'head', ' dep', 'right' ;

printf OUT "%s\n", "  -----------+-------+-------+------+-------+------+-------+-------" ;

printf OUT "  %-10s | %5d | %5d | %3.0f%% | %5d | %3.0f%% | %5d | %3.0f%%\n",
  'total', $counts{tot},
  $counts{tot}-$counts{err_head}{tot}, 100-$counts{err_head}{tot}*100.0/$counts{tot},
  $counts{tot}-$counts{err_dep}{tot}, 100-$counts{err_dep}{tot}*100.0/$counts{tot},
  $counts{tot}-$counts{err_any}, 100-$counts{err_any}*100.0/$counts{tot} ;

printf OUT "%s\n", "  -----------+-------+-------+------+-------+------+-------+-------" ;

foreach $pos (sort {$counts{pos}{$b}{tot} <=> $counts{pos}{$a}{tot}} keys %{$counts{pos}})
{
    if (! defined($counts{pos}{$pos}{err_head}{tot}))
    {
	$counts{pos}{$pos}{err_head}{tot} = 0 ;
    }
    if (! defined($counts{pos}{$pos}{err_dep}{tot}))
    {
	$counts{pos}{$pos}{err_dep}{tot} = 0 ;
    }
    if (! defined($counts{pos}{$pos}{err_any}))
    {
	$counts{pos}{$pos}{err_any} = 0 ;
    }

    printf OUT "  %-10s | %5d | %5d | %3.0f%% | %5d | %3.0f%% | %5d | %3.0f%%\n",
    $pos, $counts{pos}{$pos}{tot},
    $counts{pos}{$pos}{tot}-$counts{pos}{$pos}{err_head}{tot}, 100-$counts{pos}{$pos}{err_head}{tot}*100.0/$counts{pos}{$pos}{tot},
    $counts{pos}{$pos}{tot}-$counts{pos}{$pos}{err_dep}{tot}, 100-$counts{pos}{$pos}{err_dep}{tot}*100.0/$counts{pos}{$pos}{tot},
    $counts{pos}{$pos}{tot}-$counts{pos}{$pos}{err_any}, 100-$counts{pos}{$pos}{err_any}*100.0/$counts{pos}{$pos}{tot} ;
}

printf OUT "%s\n", "  -----------+-------+-------+------+-------+------+-------+-------" ;

printf OUT "\n\n" ;

printf OUT "  The overall error rate and its distribution over CPOSTAGs\n\n" ;
printf OUT "%s\n", "  -----------+-------+-------+------+-------+------+-------+-------" ;

printf OUT "  %-10s | %-5s | %-5s |   %%  | %-5s |   %%  | %-5s |   %%\n",
  'Error', 'words', 'head', ' dep', 'both' ;
printf OUT "  %-10s | %-5s | %-5s |      | %-5s |      | %-5s |\n",

  'Rate', ' ', 'err', ' err', 'wrong' ;

printf OUT "%s\n", "  -----------+-------+-------+------+-------+------+-------+-------" ;

printf OUT "  %-10s | %5d | %5d | %3.0f%% | %5d | %3.0f%% | %5d | %3.0f%%\n",
  'total', $counts{tot},
  $counts{err_head}{tot}, $counts{err_head}{tot}*100.0/$counts{tot},
  $counts{err_dep}{tot}, $counts{err_dep}{tot}*100.0/$counts{tot},
  $counts{err_both}, $counts{err_both}*100.0/$counts{tot} ;

printf OUT "%s\n", "  -----------+-------+-------+------+-------+------+-------+-------" ;

foreach $pos (sort {$counts{pos}{$b}{tot} <=> $counts{pos}{$a}{tot}} keys %{$counts{pos}})
{
    if (! defined($counts{pos}{$pos}{err_both}))
    {
	$counts{pos}{$pos}{err_both} = 0 ;
    }

    printf OUT "  %-10s | %5d | %5d | %3.0f%% | %5d | %3.0f%% | %5d | %3.0f%%\n",
    $pos, $counts{pos}{$pos}{tot},
    $counts{pos}{$pos}{err_head}{tot}, $counts{pos}{$pos}{err_head}{tot}*100.0/$counts{pos}{$pos}{tot},
    $counts{pos}{$pos}{err_dep}{tot}, $counts{pos}{$pos}{err_dep}{tot}*100.0/$counts{pos}{$pos}{tot},
    $counts{pos}{$pos}{err_both}, $counts{pos}{$pos}{err_both}*100.0/$counts{pos}{$pos}{tot} ;
    
}

printf OUT "%s\n", "  -----------+-------+-------+------+-------+------+-------+-------" ;

### added by Sabine Buchholz
printf OUT "\n\n";
printf OUT "  Precision and recall of DEPREL\n\n";
printf OUT "  ----------------+------+---------+--------+------------+---------------\n";
printf OUT "  deprel          | gold | correct | system | recall (%%) | precision (%%) \n";
printf OUT "  ----------------+------+---------+--------+------------+---------------\n";
foreach my $dep (sort keys %{$counts{all_dep}}) {
    # initialize
    my ($tot_corr, $tot_g, $tot_s, $prec, $rec) = (0, 0, 0, 'NaN', 'NaN');

    if (defined($counts{dep2}{$dep}{$dep})) {
	$tot_corr = $counts{dep2}{$dep}{$dep};
    } 
    if (defined($counts{dep}{$dep}{tot})) {
    	$tot_g = $counts{dep}{$dep}{tot};
	$rec = sprintf("%.2f",$tot_corr / $tot_g * 100);
    }
    if (defined($counts{dep_s}{$dep}{tot})) {
	$tot_s = $counts{dep_s}{$dep}{tot};
	$prec = sprintf("%.2f",$tot_corr / $tot_s * 100);
    }
    printf OUT "  %-15s | %4d | %7d | %6d | %10s | %13s\n",
    $dep, $tot_g, $tot_corr, $tot_s, $rec, $prec;
}

### DEPREL + ATTACHMENT:
### Same as Sabine's DEPREL apart from $tot_corr calculation
printf OUT "\n\n";
printf OUT "  Precision and recall of DEPREL + ATTACHMENT\n\n";
printf OUT "  ----------------+------+---------+--------+------------+---------------\n";
printf OUT "  deprel          | gold | correct | system | recall (%%) | precision (%%) \n";
printf OUT "  ----------------+------+---------+--------+------------+---------------\n";
foreach my $dep (sort keys %{$counts{all_dep}}) {
    # initialize
    my ($tot_corr, $tot_g, $tot_s, $prec, $rec) = (0, 0, 0, 'NaN', 'NaN');

    if (defined($counts{dep2}{$dep}{$dep})) {
	if (defined($counts{err_head_corr_dep}{$dep})) {
	    $tot_corr = $counts{dep2}{$dep}{$dep} - $counts{err_head_corr_dep}{$dep};
	} else {
	    $tot_corr = $counts{dep2}{$dep}{$dep};
	}
    } 
    if (defined($counts{dep}{$dep}{tot})) {
    	$tot_g = $counts{dep}{$dep}{tot};
	$rec = sprintf("%.2f",$tot_corr / $tot_g * 100);
    }
    if (defined($counts{dep_s}{$dep}{tot})) {
	$tot_s = $counts{dep_s}{$dep}{tot};
	$prec = sprintf("%.2f",$tot_corr / $tot_s * 100);
    }
    printf OUT "  %-15s | %4d | %7d | %6d | %10s | %13s\n",
    $dep, $tot_g, $tot_corr, $tot_s, $rec, $prec;
}
### DEPREL + ATTACHMENT

printf OUT "\n\n";
printf OUT "  Precision and recall of binned HEAD direction\n\n";
printf OUT "  ----------------+------+---------+--------+------------+---------------\n";
printf OUT "  direction       | gold | correct | system | recall (%%) | precision (%%) \n";
printf OUT "  ----------------+------+---------+--------+------------+---------------\n";
foreach my $dir ('to_root', 'left', 'right', 'self') {
    # initialize
    my ($tot_corr, $tot_g, $tot_s, $prec, $rec) = (0, 0, 0, 'NaN', 'NaN');

    if (defined($counts{dir2}{$dir}{$dir})) {
	$tot_corr = $counts{dir2}{$dir}{$dir};
    } 
    if (defined($counts{dir_g}{$dir}{tot})) {
    	$tot_g = $counts{dir_g}{$dir}{tot};
	$rec = sprintf("%.2f",$tot_corr / $tot_g * 100);
    }
    if (defined($counts{dir_s}{$dir}{tot})) {
	$tot_s = $counts{dir_s}{$dir}{tot};
	$prec = sprintf("%.2f",$tot_corr / $tot_s * 100);
    }
    printf OUT "  %-15s | %4d | %7d | %6d | %10s | %13s\n",
    $dir, $tot_g, $tot_corr, $tot_s, $rec, $prec;
}

printf OUT "\n\n";
printf OUT "  Precision and recall of binned HEAD distance\n\n";
printf OUT "  ----------------+------+---------+--------+------------+---------------\n";
printf OUT "  distance        | gold | correct | system | recall (%%) | precision (%%) \n";
printf OUT "  ----------------+------+---------+--------+------------+---------------\n";
foreach my $dist ('to_root', '1', '2', '3-6', '7-...') {
    # initialize
    my ($tot_corr, $tot_g, $tot_s, $prec, $rec) = (0, 0, 0, 'NaN', 'NaN');

    if (defined($counts{dist2}{$dist}{$dist})) {
	$tot_corr = $counts{dist2}{$dist}{$dist};
    } 
    if (defined($counts{dist_g}{$dist}{tot})) {
    	$tot_g = $counts{dist_g}{$dist}{tot};
	$rec = sprintf("%.2f",$tot_corr / $tot_g * 100);
    }
    if (defined($counts{dist_s}{$dist}{tot})) {
	$tot_s = $counts{dist_s}{$dist}{tot};
	$prec = sprintf("%.2f",$tot_corr / $tot_s * 100);
    }
    printf OUT "  %-15s | %4d | %7d | %6d | %10s | %13s\n",
    $dist, $tot_g, $tot_corr, $tot_s, $rec, $prec;
}

printf OUT "\n\n";
printf OUT "  Frame confusions (gold versus system; *...* marks the head token)\n\n";
foreach my $frame (sort {$counts{frame2}{$b} <=> $counts{frame2}{$a}} keys %{$counts{frame2}})
{
    if ($counts{frame2}{$frame} >= 5) # (make 5 a changeable threshold later)
    {
	printf OUT "  %3d  %s\n", $counts{frame2}{$frame}, $frame;
    }
}
### end of: added by Sabine Buchholz


#
# Leave only the 5 words mostly involved in errors
#


$thresh = (sort {$b <=> $a} values %{$counts{word}{err_any}})[4] ;

# ensure enough space for title
$max_word_len = length('word') ;

foreach $word (keys %{$counts{word}{err_any}})
{
  if ($counts{word}{err_any}{$word} < $thresh)
  {
    delete $counts{word}{err_any}{$word} ;
    next ;
  }

  $l = uni_len($word) ;
  if ($l > $max_word_len)
  {
    $max_word_len = $l ;
  }
}

# filter a case when the difference between the error counts
# for 2-word and 1-word contexts is small
# (leave the 2-word context)

foreach $con (keys %{$counts{con_aft_2}{tot}})
{
  ($w1) = split(/\+/, $con) ;
  
  if (defined $counts{con_aft}{tot}{$w1} &&
      $counts{con_aft}{tot}{$w1}-$counts{con_aft_2}{tot}{$con} <= 1)
  {
    delete $counts{con_aft}{tot}{$w1} ;
  }
}

foreach $con (keys %{$counts{con_bef_2}{tot}})
{
  ($w_2, $w_1) = split(/\+/, $con) ;

  if (defined $counts{con_bef}{tot}{$w_1} &&
      $counts{con_bef}{tot}{$w_1}-$counts{con_bef_2}{tot}{$con} <= 1)
  {
    delete $counts{con_bef}{tot}{$w_1} ;
  }
}

foreach $con_pos (keys %{$counts{con_pos_aft_2}{tot}})
{
  ($p1) = split(/\+/, $con_pos) ;

  if (defined($counts{con_pos_aft}{tot}{$p1}) &&
      $counts{con_pos_aft}{tot}{$p1}-$counts{con_pos_aft_2}{tot}{$con_pos} <= 1)
  {
    delete $counts{con_pos_aft}{tot}{$p1} ;
  }
}

foreach $con_pos (keys %{$counts{con_pos_bef_2}{tot}})
{
  ($p_2, $p_1) = split(/\+/, $con_pos) ;

  if (defined($counts{con_pos_bef}{tot}{$p_1}) &&
      $counts{con_pos_bef}{tot}{$p_1}-$counts{con_pos_bef_2}{tot}{$con_pos} <= 1)
  {
    delete $counts{con_pos_bef}{tot}{$p_1} ;
  }
}

# for each context type, take the three contexts most involved in errors

$max_con_len = 0 ;

filter_context_counts($counts{con_bef_2}{tot}, $con_err_num, \$max_con_len) ;

filter_context_counts($counts{con_bef}{tot}, $con_err_num, \$max_con_len) ;

filter_context_counts($counts{con_aft}{tot}, $con_err_num, \$max_con_len) ;

filter_context_counts($counts{con_aft_2}{tot}, $con_err_num, \$max_con_len) ;

# for each CPOS context type, take the three CPOS contexts most involved in errors

$max_con_pos_len = 0 ;

$thresh = (sort {$b <=> $a} values %{$counts{con_pos_bef_2}{tot}})[$con_err_num-1] ;

foreach $con_pos (keys %{$counts{con_pos_bef_2}{tot}})
{
  if ($counts{con_pos_bef_2}{tot}{$con_pos} < $thresh)
  {
    delete $counts{con_pos_bef_2}{tot}{$con_pos} ;
    next ;
  }
  if (length($con_pos) > $max_con_pos_len)
  {
    $max_con_pos_len = length($con_pos) ;
  }
}

$thresh = (sort {$b <=> $a} values %{$counts{con_pos_bef}{tot}})[$con_err_num-1] ;

foreach $con_pos (keys %{$counts{con_pos_bef}{tot}})
{
  if ($counts{con_pos_bef}{tot}{$con_pos} < $thresh)
  {
    delete $counts{con_pos_bef}{tot}{$con_pos} ;
    next ;
  }
  if (length($con_pos) > $max_con_pos_len)
  {
    $max_con_pos_len = length($con_pos) ;
  }
}

$thresh = (sort {$b <=> $a} values %{$counts{con_pos_aft}{tot}})[$con_err_num-1] ;

foreach $con_pos (keys %{$counts{con_pos_aft}{tot}})
{
  if ($counts{con_pos_aft}{tot}{$con_pos} < $thresh)
  {
    delete $counts{con_pos_aft}{tot}{$con_pos} ;
    next ;
  }
  if (length($con_pos) > $max_con_pos_len)
  {
    $max_con_pos_len = length($con_pos) ;
  }
}

$thresh = (sort {$b <=> $a} values %{$counts{con_pos_aft_2}{tot}})[$con_err_num-1] ;

foreach $con_pos (keys %{$counts{con_pos_aft_2}{tot}})
{
  if ($counts{con_pos_aft_2}{tot}{$con_pos} < $thresh)
  {
    delete $counts{con_pos_aft_2}{tot}{$con_pos} ;
    next ;
  }
  if (length($con_pos) > $max_con_pos_len)
  {
    $max_con_pos_len = length($con_pos) ;
  }
}

# printing

# ------------- focus words

printf OUT "\n\n" ;
printf OUT "  %d focus words where most of the errors occur:\n\n", scalar keys %{$counts{word}{err_any}} ;

printf OUT "  %-*s | %-4s | %-4s | %-4s | %-4s\n", $max_word_len, ' ', 'any', 'head', 'dep', 'both' ;
printf OUT "  %s-+------+------+------+------\n", '-' x $max_word_len;

foreach $word (sort {$counts{word}{err_any}{$b} <=> $counts{word}{err_any}{$a}} keys %{$counts{word}{err_any}})
{
    if (!defined($counts{word}{err_head}{$word}))
    {
	$counts{word}{err_head}{$word} = 0 ;
    }
    if (! defined($counts{word}{err_dep}{$word}))
    {
	$counts{word}{err_dep}{$word} = 0 ;
    }
    if (! defined($counts{word}{err_any}{$word}))
    {
	$counts{word}{err_any}{$word} = 0;
    }
    printf OUT "  %-*s | %4d | %4d | %4d | %4d\n",
    $max_word_len+length($word)-uni_len($word), $word, $counts{word}{err_any}{$word},
    $counts{word}{err_head}{$word},
    $counts{word}{err_dep}{$word},
    $counts{word}{err_dep}{$word}+$counts{word}{err_head}{$word}-$counts{word}{err_any}{$word} ;
}

printf OUT "  %s-+------+------+------+------\n", '-' x $max_word_len;

# ------------- contexts

printf OUT "\n\n" ;

printf OUT "  one-token preceeding contexts where most of the errors occur:\n\n" ;

print_context($counts{con_bef}, $counts{con_pos_bef}, $max_con_len, $max_con_pos_len) ;

printf OUT "  two-token preceeding contexts where most of the errors occur:\n\n" ;

print_context($counts{con_bef_2}, $counts{con_pos_bef_2}, $max_con_len, $max_con_pos_len) ;

printf OUT "  one-token following contexts where most of the errors occur:\n\n" ;

print_context($counts{con_aft}, $counts{con_pos_aft}, $max_con_len, $max_con_pos_len) ;

printf OUT "  two-token following contexts where most of the errors occur:\n\n" ;

print_context($counts{con_aft_2}, $counts{con_pos_aft_2}, $max_con_len, $max_con_pos_len) ;

# ------------- Sentences

printf OUT "  Sentence with the highest number of word errors:\n" ;
$i = (sort { (defined($err_sent[$b]{word}) && $err_sent[$b]{word})
		 <=> (defined($err_sent[$a]{word}) && $err_sent[$a]{word}) } 1 .. $sent_num)[0] ;
printf OUT "   Sentence %d line %d, ", $i, $starts[$i-1] ;
printf OUT "%d head errors, %d dependency errors, %d word errors\n",
  $err_sent[$i]{head}, $err_sent[$i]{dep}, $err_sent[$i]{word} ;

printf OUT "\n\n" ;

printf OUT "  Sentence with the highest number of head errors:\n" ;
$i = (sort { (defined($err_sent[$b]{head}) && $err_sent[$b]{head}) 
		 <=> (defined($err_sent[$a]{head}) && $err_sent[$a]{head}) } 1 .. $sent_num)[0] ;
printf OUT "   Sentence %d line %d, ", $i, $starts[$i-1] ;
printf OUT "%d head errors, %d dependency errors, %d word errors\n",
  $err_sent[$i]{head}, $err_sent[$i]{dep}, $err_sent[$i]{word} ;

printf OUT "\n\n" ;

printf OUT "  Sentence with the highest number of dependency errors:\n" ;
$i = (sort { (defined($err_sent[$b]{dep}) && $err_sent[$b]{dep}) 
		 <=> (defined($err_sent[$a]{dep}) && $err_sent[$a]{dep}) } 1 .. $sent_num)[0] ;
printf OUT "   Sentence %d line %d, ", $i, $starts[$i-1] ;
printf OUT "%d head errors, %d dependency errors, %d word errors\n",
  $err_sent[$i]{head}, $err_sent[$i]{dep}, $err_sent[$i]{word} ;

#
# Second pass, collect statistics of the frequent errors
#

# filter the errors, leave the most frequent $freq_err_num errors

$i = 0 ;

$thresh = (sort {$b <=> $a} values %freq_err)[$freq_err_num-1] ;

foreach $err (keys %freq_err)
{
  if ($freq_err{$err} < $thresh)
  {
    delete $freq_err{$err} ;
  }
}

# in case there are several errors with the threshold count

$freq_err_num = scalar keys %freq_err ;

%err_counts = () ;

$eof = 0 ;

seek (GOLD, 0, 0) ;
seek (SYS, 0, 0) ;

while (! $eof)
{ # second reading loop

  $eof = read_sent(\@sent_gold, \@sent_sys) ;
  $sent_num++ ;

  $word_num = scalar @sent_gold ;

  # printf "$sent_num $word_num\n" ;
  
  foreach $i_w (0 .. $word_num-1)
  { # loop on words
    ($word, $pos, $head_g, $dep_g)
      = @{$sent_gold[$i_w]}{'word', 'pos', 'head', 'dep'} ;

    # printf "%d: %s %s %s %s\n", $i_w,  $word, $pos, $head_g, $dep_g ;

    if ((! $score_on_punct) && is_uni_punct($word))
    {
      # ignore punctuations
      next ;
    }

    ($head_s, $dep_s) = @{$sent_sys[$i_w]}{'head', 'dep'} ;

    $err_head = ($head_g ne $head_s) ;
    $err_dep = ($dep_g ne $dep_s) ;

    $head_err = '-' ;
    $dep_err = '-' ;

    if ($head_g eq '0')
    {
      $head_aft_bef_g = '0' ;
    }
    elsif ($head_g eq $i_w+1)
    {
      $head_aft_bef_g = 'e' ;
    }
    else
    {
      $head_aft_bef_g = ($head_g <= $i_w+1 ? 'b' : 'a') ;
    }

    if ($head_s eq '0')
    {
      $head_aft_bef_s = '0' ;
    }
    elsif ($head_s eq $i_w+1)
    {
      $head_aft_bef_s = 'e' ;
    }
    else
    {
      $head_aft_bef_s = ($head_s <= $i_w+1 ? 'b' : 'a') ;
    }

    $head_aft_bef = $head_aft_bef_g.$head_aft_bef_s ;

    if ($err_head)
    {
      if ($head_aft_bef_s eq '0')
      {
	$head_err = 0 ;
      }
      else
      {
	$head_err = $head_s-$head_g ;
      }
    }

    if ($err_dep)
    {
      $dep_err = $dep_g.'->'.$dep_s ;
    }

    if (! ($err_head || $err_dep))
    {
      next ;
    }

    # handle only the most frequent errors

    $err = $head_err.$sep.$head_aft_bef.$sep.$dep_err ;

    if (! exists $freq_err{$err})
    {
      next ;
    }

    ($w_2, $w_1, $w1, $w2, $p_2, $p_1, $p1, $p2) = get_context(\@sent_gold, $i_w) ;

    $con_bef = $w_1 ;
    $con_bef_2 = $w_2.' + '.$w_1 ;
    $con_aft = $w1 ;
    $con_aft_2 = $w1.' + '.$w2 ;

    $con_pos_bef = $p_1 ;
    $con_pos_bef_2 = $p_2.'+'.$p_1 ;
    $con_pos_aft = $p1 ;
    $con_pos_aft_2 = $p1.'+'.$p2 ;

    @cur_err = ($con_pos_bef, $con_bef, $word, $pos, $con_pos_aft, $con_aft) ;

    # printf "# %-25s %-15s %-10s %-25s %-3s %-30s\n",
    #  $con_bef, $word, $pos, $con_aft, $head_err, $dep_err ;
    
    @bits = (0, 0, 0, 0, 0, 0) ;
    $j = 0 ;

    while ($j == 0)
    {
      for ($i = 0; $i <= $#bits; $i++)
      {
	if ($bits[$i] == 0)
	{
	  $bits[$i] = 1 ;
	  $j = 0 ;
	  last ;
	}
	else
	{
	  $bits[$i] = 0 ;
	  $j = 1 ;
	}
      }

      @e_bits = @cur_err ;

      for ($i = 0; $i <= $#bits; $i++)
      {
	if (! $bits[$i])
	{
	  $e_bits[$i] = '*' ;
	}
      }

      # include also the last case which is the most general
      # (wildcards for everything)
      $err_counts{$err}{join($sep, @e_bits)}++ ;

    }

  } # loop on words
} # second reading loop

printf OUT "\n\n" ;
printf OUT "  Specific errors, %d most frequent errors:", $freq_err_num ;
printf OUT "\n  %s\n", '=' x 41 ;


# deleting local contexts which are too general

foreach $err (keys %err_counts)
{
  foreach $loc_con (sort {$err_counts{$err}{$b} <=> $err_counts{$err}{$a}}
		    keys %{$err_counts{$err}})
  {
    @cur_err = split(/\Q$sep\E/, $loc_con) ;

    # In this loop, one or two elements of the local context are
    # replaced with '*' to make it more general. If the entry for
    # the general context has the same count it is removed.

    foreach $i (0 .. $#cur_err)
    {
      $w1 = $cur_err[$i] ;
      if ($cur_err[$i] eq '*')
      {
	next ;
      }
      $cur_err[$i] = '*' ;
      $con1 = join($sep, @cur_err) ;
      if ( defined($err_counts{$err}{$con1}) && defined($err_counts{$err}{$loc_con})
	   && ($err_counts{$err}{$con1} == $err_counts{$err}{$loc_con}))
      {
	delete $err_counts{$err}{$con1} ;
      }
      for ($j = $i+1; $j <=$#cur_err; $j++)
      {
	if ($cur_err[$j] eq '*')
	{
	  next ;
	}
	$w2 = $cur_err[$j] ;
	$cur_err[$j] = '*' ;
	$con1 = join($sep, @cur_err) ;
	if ( defined($err_counts{$err}{$con1}) && defined($err_counts{$err}{$loc_con})
	     && ($err_counts{$err}{$con1} == $err_counts{$err}{$loc_con}))
	{
	  delete $err_counts{$err}{$con1} ;
	}
	$cur_err[$j] = $w2 ;
      }
      $cur_err[$i] = $w1 ;
    }
  }
}

# Leaving only the topmost local contexts for each error

foreach $err (keys %err_counts)
{
  $thresh = (sort {$b <=> $a} values %{$err_counts{$err}})[$spec_err_loc_con-1] || 0 ;

  # of the threshold is too low, take the 2nd highest count
  # (the highest may be the total which is the generic case
  #   and not relevant for printing)

  if ($thresh < 5)
  {
    $thresh = (sort {$b <=> $a} values %{$err_counts{$err}})[1] ;
  }

  foreach $loc_con (keys %{$err_counts{$err}})
  {
    if ($err_counts{$err}{$loc_con} < $thresh)
    {
      delete $err_counts{$err}{$loc_con} ;
    }
    else
    {
      if ($loc_con ne join($sep, ('*', '*', '*', '*', '*', '*')))
      {
	$loc_con_err_counts{$loc_con}{$err} = $err_counts{$err}{$loc_con} ;
      }
    }
  }
}

# printing an error summary

# calculating the context field length

$max_word_spec_len= length('word') ;
$max_con_aft_len = length('word') ;
$max_con_bef_len = length('word') ;
$max_con_pos_len = length('CPOS') ;

foreach $err (keys %err_counts)
{
  foreach $loc_con (sort keys %{$err_counts{$err}})
  {
    ($con_pos_bef, $con_bef, $word, $pos, $con_pos_aft, $con_aft) =
      split(/\Q$sep\E/, $loc_con) ;

    $l = uni_len($word) ;
    if ($l > $max_word_spec_len)
    {
      $max_word_spec_len = $l ;
    }

    $l = uni_len($con_bef) ;
    if ($l > $max_con_bef_len)
    {
      $max_con_bef_len = $l ;
    }

    $l = uni_len($con_aft) ;
    if ($l > $max_con_aft_len)
    {
      $max_con_aft_len = $l ;
    }

    if (length($con_pos_aft) > $max_con_pos_len)
    {
      $max_con_pos_len = length($con_pos_aft) ;
    }

    if (length($con_pos_bef) > $max_con_pos_len)
    {
      $max_con_pos_len = length($con_pos_bef) ;
    }
  }
}

$err_counter = 0 ;

foreach $err (sort {$freq_err{$b} <=> $freq_err{$a}} keys %freq_err)
{

  ($head_err, $head_aft_bef, $dep_err) = split(/\Q$sep\E/, $err) ;

  $err_counter++ ;
  $err_desc{$err} = sprintf("%2d. ", $err_counter).
    describe_err($head_err, $head_aft_bef, $dep_err) ;
  
  # printf OUT "  %-3s %-30s %d\n", $head_err, $dep_err, $freq_err{$err} ;
  printf OUT "\n" ;
  printf OUT "  %s : %d times\n", $err_desc{$err}, $freq_err{$err} ;

  printf OUT "  %s-+-%s-+-%s-+-%s-+-%s-+-%s-+------\n",
    '-' x $max_con_pos_len, '-' x $max_con_bef_len,
       '-' x $max_pos_len, '-' x $max_word_spec_len,
	'-' x $max_con_pos_len, '-' x $max_con_aft_len ;

  printf OUT "  %-*s | %-*s | %-*s | %s\n",
      $max_con_pos_len+$max_con_bef_len+3, '  Before',
	$max_word_spec_len+$max_pos_len+3, '   Focus',
	  $max_con_pos_len+$max_con_aft_len+3, '  After',
	    'Count' ;

  printf OUT "  %-*s   %-*s | %-*s   %-*s | %-*s   %-*s |\n",
    $max_con_pos_len, 'CPOS', $max_con_bef_len, 'word',
       $max_pos_len, 'CPOS', $max_word_spec_len, 'word',
	$max_con_pos_len, 'CPOS', $max_con_aft_len, 'word' ;
  
  printf OUT "  %s-+-%s-+-%s-+-%s-+-%s-+-%s-+------\n",
    '-' x $max_con_pos_len, '-' x $max_con_bef_len,
       '-' x $max_pos_len, '-' x $max_word_spec_len,
	'-' x $max_con_pos_len, '-' x $max_con_aft_len ;

  foreach $loc_con (sort {$err_counts{$err}{$b} <=> $err_counts{$err}{$a}}
		    keys %{$err_counts{$err}})
  {
    if ($loc_con eq join($sep, ('*', '*', '*', '*', '*', '*')))
    {
      next ;
    }

    $con1 = $loc_con ;
    $con1 =~ s/\*/ /g ;

    ($con_pos_bef, $con_bef, $word, $pos, $con_pos_aft, $con_aft) =
      split(/\Q$sep\E/, $con1) ;

    printf OUT "  %-*s | %-*s | %-*s | %-*s | %-*s | %-*s | %3d\n",
      $max_con_pos_len, $con_pos_bef, $max_con_bef_len+length($con_bef)-uni_len($con_bef), $con_bef,
	  $max_pos_len, $pos, $max_word_spec_len+length($word)-uni_len($word), $word,
	    $max_con_pos_len, $con_pos_aft, $max_con_aft_len+length($con_aft)-uni_len($con_aft), $con_aft,
	      $err_counts{$err}{$loc_con} ;
  }
  
  printf OUT "  %s-+-%s-+-%s-+-%s-+-%s-+-%s-+------\n",
    '-' x $max_con_pos_len, '-' x $max_con_bef_len,
       '-' x $max_pos_len, '-' x $max_word_spec_len,
	'-' x $max_con_pos_len, '-' x $max_con_aft_len ;

}

printf OUT "\n\n" ;
printf OUT "  Local contexts involved in several frequent errors:" ;
printf OUT "\n  %s\n", '=' x 51 ;
printf OUT "\n\n" ;

foreach $loc_con (sort {scalar keys %{$loc_con_err_counts{$b}} <=>
			  scalar keys %{$loc_con_err_counts{$a}}}
		  keys %loc_con_err_counts)
{

  if (scalar keys %{$loc_con_err_counts{$loc_con}} == 1)
  {
    next ;
  }
  
  printf OUT "  %s-+-%s-+-%s-+-%s-+-%s-+-%s-\n",
    '-' x $max_con_pos_len, '-' x $max_con_bef_len,
       '-' x $max_pos_len, '-' x $max_word_spec_len,
	'-' x $max_con_pos_len, '-' x $max_con_aft_len ;

  printf OUT "  %-*s | %-*s | %-*s \n",
      $max_con_pos_len+$max_con_bef_len+3, '  Before',
	$max_word_spec_len+$max_pos_len+3, '   Focus',
	  $max_con_pos_len+$max_con_aft_len+3, '  After' ;

  printf OUT "  %-*s   %-*s | %-*s   %-*s | %-*s   %-*s \n",
    $max_con_pos_len, 'CPOS', $max_con_bef_len, 'word',
       $max_pos_len, 'CPOS', $max_word_spec_len, 'word',
	$max_con_pos_len, 'CPOS', $max_con_aft_len, 'word' ;
  
  printf OUT "  %s-+-%s-+-%s-+-%s-+-%s-+-%s-\n",
    '-' x $max_con_pos_len, '-' x $max_con_bef_len,
       '-' x $max_pos_len, '-' x $max_word_spec_len,
	'-' x $max_con_pos_len, '-' x $max_con_aft_len ;

  $con1 = $loc_con ;
  $con1 =~ s/\*/ /g ;

  ($con_pos_bef, $con_bef, $word, $pos, $con_pos_aft, $con_aft) =
      split(/\Q$sep\E/, $con1) ;

  printf OUT "  %-*s | %-*s | %-*s | %-*s | %-*s | %-*s \n",
    $max_con_pos_len, $con_pos_bef, $max_con_bef_len+length($con_bef)-uni_len($con_bef), $con_bef,
      $max_pos_len, $pos, $max_word_spec_len+length($word)-uni_len($word), $word,
	$max_con_pos_len, $con_pos_aft, $max_con_aft_len+length($con_aft)-uni_len($con_aft), $con_aft ;
	  
  printf OUT "  %s-+-%s-+-%s-+-%s-+-%s-+-%s-\n",
    '-' x $max_con_pos_len, '-' x $max_con_bef_len,
       '-' x $max_pos_len, '-' x $max_word_spec_len,
	'-' x $max_con_pos_len, '-' x $max_con_aft_len ;

  foreach $err (sort {$loc_con_err_counts{$loc_con}{$b} <=>
			$loc_con_err_counts{$loc_con}{$a}}
		keys %{$loc_con_err_counts{$loc_con}})
  {
    printf OUT "  %s : %d times\n", $err_desc{$err},
      $loc_con_err_counts{$loc_con}{$err} ;
  }

  printf OUT "\n" ;
}

close GOLD ;
close SYS ;

close OUT ;
