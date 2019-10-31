program solitaire
implicit none

! Auxiliary variables
integer :: i, j
character (len = 5) :: card_format = '(A18)'

! Card names for I/O
character (len = 6) :: values(13) = (/'ACE   ', 'TWO   ', 'THREE ', &
  'FOUR  ', 'FIVE  ', 'SIX   ', 'SEVEN ', 'EIGHT ', 'NINE  ', 'TEN   ', &
  'JACK  ', 'QUEEN ', 'KING  '/)
character (len = 9) :: suits(4) = (/'CLUBS    ', 'HEARTS   ', &
  'SPADES   ', 'DIAMONDS ' /)

! Card Structures
type :: Card
  integer :: val = 0
  integer :: suit = 0
  logical :: visible = .false.
end type Card

type Pile
  type (Card), dimension (20) :: cards
  integer :: used = 0
end type Pile

type DrawDeck
  type (Card), dimension (52) :: cards
  integer :: used = 52
  integer :: current = 52
end type DrawDeck

! Game State
type (Pile), dimension(4) :: stacks
type (Pile), dimension(7) :: piles
type (DrawDeck) :: deck

! Initialize Game State

! Fill the deck
do i=1,52
  deck%cards(i)%val = modulo(i, 13) + 1
  deck%cards(i)%suit = modulo(i, 4) + 1
end do

! Shuffle the deck

! Fill the piles
do i=1,7
  piles(i)%used = i
  do j=1,i
    call set(piles(i), pop(deck), j)
    if (j == i) then
      piles(i)%cards(j)%visible = .true.
    end if
  end do
end do

call printMain()

contains
  ! Card Structure Operations

  function pop(deck) result(res)
    type (DrawDeck) :: deck
    type (Card) :: res
    
    res%val = deck%cards(deck%used)%val
    res%suit = deck%cards(deck%used)%suit
    res%visible = deck%cards(deck%used)%visible

    deck%cards(deck%used)%val = 0
    deck%cards(deck%used)%suit = 0
    deck%cards(deck%used)%visible = .false.

    deck%used = deck%used - 1
  end function pop

  function peek(stack) result(res)
    type (Pile) :: stack
    type (Card) :: res

    res%val = stack%cards(stack%used)%val
    res%suit = stack%cards(stack%used)%suit
    res%visible = stack%cards(stack%used)%visible
  end function peek

  subroutine set(stack, cardval, cardidx)
    type (Pile) :: stack
    type (Card) :: cardval
    integer :: cardidx

    stack%cards(cardidx)%val = cardval%val
    stack%cards(cardidx)%suit = cardval%suit
    stack%cards(cardidx)%visible = cardval%visible
  end subroutine set

  ! I/O

  subroutine printCard(cardval, newline)
    type (Card) :: cardval
    logical :: newline
    character (len = 18) :: outstr

    if (cardval%val == 0) then
      outstr = '                  '
    else if (.not. cardval%visible) then
      outstr = 'HIDDEN            '
    else
      outstr = values(cardval%val) // 'OF ' // suits(cardval%suit)
    end if

    if (newline) then
      write(*, card_format) outstr
    else
      write(*, card_format, advance='no') outstr
    end if
  end subroutine printCard

  subroutine printMain()
    do i=1,4
      call printCard(peek(stacks(i)), i == 4)
    end do

    do i=1,20
      do j=1,7
        call printCard(piles(j)%cards(i), j == 7)
      end do
    end do
  end subroutine printMain

end program solitaire

